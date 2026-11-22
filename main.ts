// A Sokoban solver for make.fun's sokoban-solver-01: push-optimal per room.
//
// Strategy: the level is a hallway with 1-4 rooms off it, and a box can never
// leave its room, so the level splits into independent rooms (cut the hallway
// row's dead squares, flood-fill what is left). Each room is solved on its own
// with A* over pushes: a state is the boxes plus the player's reachable region,
// the estimate is the cheapest assignment of boxes to goals over walking
// distances, and two deadlocks are pruned (a box on a square no pull can bring
// to a goal, and a frozen 2x2 block). A greedy search and then a plain
// breadth-first search stand by for any room the A* gives up on, so an
// instance is only ever left unsolved when no search in its budget finds a
// way. Everything is budgeted by work (fixed expansion caps), never by the
// clock, so two runs write the same bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// [dx, dy, step letter, push letter]
const DIRS = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
] as const;

const ASTAR_CAP = 400_000;
const GREEDY_CAP = 200_000;
const BFS_CAP = 250_000;

type Board = {
  w: number;
  h: number;
  n: number;
  wall: Uint8Array;
  goals: number[];
  goalSet: Uint8Array;
  boxes: number[];
  player: number;
};

/** Parse an XSB level: # wall, space floor, . goal, $ box, * box on goal, @ player, + player on goal. */
function parseLevel(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const n = w * h;
  const wall = new Uint8Array(n).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  const goalSet = new Uint8Array(n);
  for (const g of goals) goalSet[g] = 1;
  return { w, h, n, wall, goals: goals.sort((a, b) => a - b), goalSet, boxes: boxes.sort((a, b) => a - b), player };
}

/** Squares a box alone can be pulled onto some goal (reverse pulls from the goals). */
function liveSquares(b: Board): Uint8Array {
  const live = new Uint8Array(b.n);
  const queue: number[] = [];
  for (const g of b.goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let i = 0; i < queue.length; i++) {
    const x = queue[i];
    for (const [dx, dy] of DIRS) {
      const d = dy * b.w + dx;
      const y = x - d;
      const z = y - d;
      if (y < 0 || y >= b.n || z < 0 || z >= b.n) continue;
      if (!b.wall[y] && !b.wall[z] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** Lowest square reachable from `from` around walls and boxes; marks the region in `seen` with `stamp`. */
function floodLow(b: Board, occupied: Uint8Array, from: number, seen: Int32Array, stamp: number, stack: Int32Array): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (const [dx, dy] of DIRS) {
      const q = p + dy * b.w + dx;
      if (q < 0 || q >= b.n) continue;
      if (!b.wall[q] && !occupied[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
function frozenBlock(b: Board, occupied: Uint8Array, t: number): boolean {
  const w = b.w;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const c1 = a + 1;
    const c2 = a + w;
    const c3 = c2 + 1;
    if (a < 0 || c3 >= b.n) continue;
    const full = (c: number): boolean => b.wall[c] === 1 || occupied[c] === 1;
    if (!full(a) || !full(c1) || !full(c2) || !full(c3)) continue;
    if ((occupied[a] && !b.goalSet[a]) || (occupied[c1] && !b.goalSet[c1]) || (occupied[c2] && !b.goalSet[c2]) || (occupied[c3] && !b.goalSet[c3])) return true;
  }
  return false;
}

/** Whether box `x`, off its goal, can be pushed in no direction at all no matter how the other boxes move: every way is shut by a wall (walls never move, so this never prunes a solvable state). */
function boxFrozen(b: Board, x: number): boolean {
  if (b.goalSet[x]) return false;
  for (const [dx, dy] of DIRS) {
    const d = dy * b.w + dx;
    const behind = x - d;
    const ahead = x + d;
    if (behind < 0 || ahead >= b.n) continue;
    if (!b.wall[behind] && !b.wall[ahead]) return false;
  }
  return true;
}

/** Walking distances from `from` over open floor (walls only), -1 where unreachable. */
function bfsDist(b: Board, from: number): Int32Array {
  const dist = new Int32Array(b.n).fill(-1);
  const queue = [from];
  dist[from] = 0;
  for (let i = 0; i < queue.length; i++) {
    const p = queue[i];
    for (const [dx, dy] of DIRS) {
      const q = p + dy * b.w + dx;
      if (q < 0 || q >= b.n || b.wall[q] || dist[q] >= 0) continue;
      dist[q] = dist[p] + 1;
      queue.push(q);
    }
  }
  return dist;
}

/** All orderings of 0..m-1, for the cheapest box-to-goal assignment. */
function permutations(m: number): number[][] {
  const out: number[][] = [];
  const a = Array.from({ length: m }, (_, i) => i);
  const gen = (k: number): void => {
    if (k === m) {
      out.push(a.slice());
      return;
    }
    for (let i = k; i < m; i++) {
      const t = a[k];
      a[k] = a[i];
      a[i] = t;
      gen(k + 1);
      a[i] = a[k];
      a[k] = t;
    }
  };
  gen(0);
  return out;
}

type Push = { from: number; to: number; dir: number };

/**
 * Push-optimal search of one board (a room on its own, or a whole level):
 * A* over pushes, `mode` 'astar' (optimal), 'greedy' (first found) or 'bfs'
 * (first found, fewest pushes). Returns the push list, or null on giving up.
 */
function searchPushes(b: Board, live: Uint8Array, startBoxes: number[], startPlayer: number, mode: 'astar' | 'greedy' | 'bfs', cap: number): Push[] | null {
  const n = b.n;
  const m = startBoxes.length;
  if (m === 0) return [];
  const dists = b.goals.map((g) => bfsDist(b, g));
  const perms = permutations(m);
  const estimate = (boxes: number[]): number => {
    let best = Infinity;
    for (const perm of perms) {
      let sum = 0;
      let ok = true;
      for (let i = 0; i < m; i++) {
        const d = dists[perm[i]]![boxes[i]!]!;
        if (d < 0) {
          ok = false;
          break;
        }
        sum += d;
      }
      if (ok && sum < best) best = sum;
    }
    return best;
  };
  const solved = (boxes: number[]): boolean => boxes.every((x) => b.goalSet[x]);
  if (startBoxes.some((x) => !live[x])) return null;
  if (solved(startBoxes)) return [];
  const occupied = new Uint8Array(n);
  for (const x of startBoxes) occupied[x] = 1;
  if (startBoxes.some((x) => frozenBlock(b, occupied, x) || boxFrozen(b, x))) return null;
  const seen = new Int32Array(n);
  const mark = new Int32Array(n);
  const stack = new Int32Array(n);
  let stamp = 0;
  const low0 = floodLow(b, occupied, startPlayer, seen, ++stamp, stack);
  for (const x of startBoxes) occupied[x] = 0;

  const keyOf = (boxes: number[], low: number): number => {
    let k = low;
    for (let i = 0; i < 4; i++) k = k * 256 + (i < boxes.length ? boxes[i]! : 255);
    return k;
  };
  const boxesArr: number[][] = [startBoxes.slice().sort((a, b) => a - b)];
  const playerArr: number[] = [startPlayer];
  const parentArr: number[] = [-1];
  const pushArr: Push[] = [{ from: -1, to: -1, dir: -1 }];
  const gArr: number[] = [0];
  const keyArr: number[] = [];
  const bestG = new Map<number, number>();
  const h0 = estimate(boxesArr[0]!);
  if (!isFinite(h0)) return null;
  keyArr.push(keyOf(boxesArr[0]!, low0));
  bestG.set(keyArr[0]!, 0);
  // Heap entries [rank1, rank2, sequence, state]: astar ranks by g+h then -g,
  // greedy by h alone, bfs by sequence alone (a plain queue).
  const heap: number[][] = [];
  const rankOf = (g: number, h: number, s: number): number[] =>
    mode === 'astar' ? [g + h, -g, s] : mode === 'greedy' ? [h, s, 0] : [s, 0, 0];
  const heapLess = (a: number[], c: number[]): boolean =>
    a[0]! < c[0]! || (a[0]! === c[0]! && (a[1]! < c[1]! || (a[1]! === c[1]! && a[2]! < c[2]!)));
  const heapPush = (e: number[]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i]!, heap[p]!)) break;
      const t = heap[i];
      heap[i] = heap[p]!;
      heap[p] = t!;
      i = p;
    }
  };
  const heapPop = (): number[] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
        if (!heapLess(heap[c]!, heap[i]!)) break;
        const t = heap[i];
        heap[i] = heap[c]!;
        heap[c] = t!;
        i = c;
      }
    }
    return top;
  };
  heapPush([...rankOf(0, h0, 0), 0]);
  let expanded = 0;
  let seq = 1;
  while (heap.length > 0) {
    const entry = heapPop();
    const s = entry[3]!;
    if (gArr[s]! > (bestG.get(keyArr[s]!) ?? Infinity)) continue;
    if (solved(boxesArr[s]!)) {
      const path: Push[] = [];
      for (let t = s; t > 0; t = parentArr[t]!) path.push(pushArr[t]!);
      path.reverse();
      return path;
    }
    if (expanded >= cap) return null;
    expanded++;
    const boxes = boxesArr[s]!;
    const g = gArr[s]!;
    for (const x of boxes) occupied[x] = 1;
    const here = ++stamp;
    floodLow(b, occupied, playerArr[s]!, seen, here, stack);
    for (let bi = 0; bi < boxes.length; bi++) {
      const box = boxes[bi]!;
      for (let di = 0; di < 4; di++) {
        const [dx, dy] = DIRS[di]!;
        const d = dy * b.w + dx;
        const behind = box - d;
        const ahead = box + d;
        if (behind < 0 || ahead >= n) continue;
        if (seen[behind] !== here || b.wall[ahead] || occupied[ahead] || !live[ahead]) continue;
        const next = boxes.slice();
        next[bi] = ahead;
        next.sort((a, z) => a - z);
        occupied[box] = 0;
        occupied[ahead] = 1;
        const dead = frozenBlock(b, occupied, ahead) || boxFrozen(b, ahead);
        let key = -1;
        let low = -1;
        let h = Infinity;
        if (!dead) {
          low = floodLow(b, occupied, box, mark, ++stamp, stack);
          key = keyOf(next, low);
          const prevG = bestG.get(key);
          if (prevG === undefined || prevG > g + 1) h = estimate(next);
        }
        occupied[ahead] = 0;
        occupied[box] = 1;
        if (dead || !isFinite(h)) continue;
        const prevG = bestG.get(key);
        if (prevG !== undefined && prevG <= g + 1) continue;
        const idx = boxesArr.length;
        boxesArr.push(next);
        playerArr.push(box);
        parentArr.push(s);
        pushArr.push({ from: box, to: ahead, dir: di });
        gArr.push(g + 1);
        keyArr.push(key);
        bestG.set(key, g + 1);
        heapPush([...rankOf(g + 1, h, seq++), idx]);
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  return null;
}

/** Shortest walk from `from` to `to` around walls and boxes, lower LURD letters. */
function walkTo(b: Board, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(b.n).fill(-1);
  const how = new Array<string>(b.n);
  const queue = [from];
  prev[from] = from;
  for (let i = 0; i < queue.length; i++) {
    const p = queue[i]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * b.w + dx;
      if (q < 0 || q >= b.n || b.wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** The player's square after playing `moves` (letters only) from `from`. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) p += ch === 'u' || ch === 'U' ? -w : ch === 'd' || ch === 'D' ? w : ch === 'l' || ch === 'L' ? -1 : 1;
  return p;
}

/** Expand a push list into the full move string from the starting position. */
function expandMoves(b: Board, startPlayer: number, startBoxes: number[], pushes: Push[]): string {
  let moves = '';
  let player = startPlayer;
  const boxes = startBoxes.slice().sort((a, z) => a - z);
  const occupied = new Uint8Array(b.n);
  for (const p of pushes) {
    for (const x of boxes) occupied[x] = 1;
    const [dx, dy, , pushLetter] = DIRS[p.dir]!;
    const d = dy * b.w + dx;
    moves += walkTo(b, occupied, player, p.from - d);
    for (const x of boxes) occupied[x] = 0;
    moves += pushLetter;
    player = p.from;
    const i = boxes.indexOf(p.from);
    boxes[i] = p.to;
    boxes.sort((a, z) => a - z);
  }
  return moves;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway:
 * a box there reaches no goal) out of the floor, and each part left holding a
 * box is a room. Null when the level is no set of rooms.
 */
function splitRooms(b: Board, live: Uint8Array): { boxes: number[]; goals: number[] }[] | null {
  const row = Math.floor(b.player / b.w);
  const cut = new Uint8Array(b.n);
  for (let x = 0; x < b.w; x++) {
    const i = row * b.w + x;
    if (!b.wall[i] && !live[i]) cut[i] = 1;
  }
  if (b.boxes.some((x) => cut[x])) return null;
  const part = new Int32Array(b.n).fill(-1);
  const rooms: { boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < b.n; s++) {
    if (b.wall[s] || cut[s] || part[s] >= 0) continue;
    const stack = [s];
    part[s] = rooms.length;
    for (let i = 0; i < stack.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = stack[i]! + dy * b.w + dx;
        if (q < 0 || q >= b.n || b.wall[q] || cut[q] || part[q] >= 0) continue;
        part[q] = rooms.length;
        stack.push(q);
      }
    }
    rooms.push({ boxes: [], goals: [] });
  }
  for (const x of b.boxes) {
    const r = part[x]!;
    if (r < 0) return null;
    rooms[r]!.boxes.push(x);
  }
  for (const g of b.goals) {
    const r = part[g]!;
    if (r < 0) return null;
    rooms[r]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Solve one room on its own board; A* first, then greedy, then breadth-first. */
function solveRoom(b: Board, boxes: number[], player: number): Push[] | null {
  const live = liveSquares(b);
  return (
    searchPushes(b, live, boxes, player, 'astar', ASTAR_CAP) ??
    searchPushes(b, live, boxes, player, 'greedy', GREEDY_CAP) ??
    searchPushes(b, live, boxes, player, 'bfs', BFS_CAP)
  );
}

/** The level's moves: room by room, the player walking on from room to room. */
function solveLevel(text: string): string {
  const b = parseLevel(text);
  const live = liveSquares(b);
  const rooms = splitRooms(b, live);
  if (rooms === null) {
    const pushes = solveRoom(b, b.boxes, b.player);
    return pushes === null ? '' : expandMoves(b, b.player, b.boxes, pushes);
  }
  let player = b.player;
  let moves = '';
  const done: boolean[] = rooms.map(() => false);
  const roomBoxes: number[][] = rooms.map((r) => r.boxes.slice());
  for (let ri = 0; ri < rooms.length; ri++) {
    const wall2 = b.wall.slice();
    for (let oj = 0; oj < rooms.length; oj++) {
      if (oj === ri) continue;
      const src = done[oj] ? rooms[oj]!.goals : roomBoxes[oj]!;
      for (const x of src) wall2[x] = 1;
    }
    const goalSet = new Uint8Array(b.n);
    for (const g of rooms[ri]!.goals) goalSet[g] = 1;
    const sub: Board = {
      w: b.w,
      h: b.h,
      n: b.n,
      wall: wall2,
      goals: rooms[ri]!.goals.slice().sort((a, z) => a - z),
      goalSet,
      boxes: roomBoxes[ri]!.slice().sort((a, z) => a - z),
      player,
    };
    const pushes = solveRoom(sub, sub.boxes, player);
    if (pushes === null) return moves;
    const chunk = expandMoves(sub, player, sub.boxes, pushes);
    moves += chunk;
    player = endOf(b.w, player, chunk);
    roomBoxes[ri] = rooms[ri]!.goals.slice();
    done[ri] = true;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
