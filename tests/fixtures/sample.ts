// Sample TypeScript fixture used by extractor_ts tests.

export interface Shape {
  area(): number;
}

export class Circle implements Shape {
  constructor(private radius: number) {}

  area(): number {
    return Math.PI * this.radius * this.radius;
  }

  describe(): string {
    return formatShape(this);
  }
}

export function formatShape(shape: Shape): string {
  return `area=${shape.area()}`;
}

export function complexFn(x: number): number {
  if (x > 0) {
    for (let i = 0; i < x; i++) {
      if (i % 2 === 0 && x > 10) {
        x = x - 1;
      } else if (i % 3 === 0 || x < 5) {
        x = x + 1;
      }
    }
  } else {
    while (x < 0) {
      x++;
    }
  }
  return x;
}
