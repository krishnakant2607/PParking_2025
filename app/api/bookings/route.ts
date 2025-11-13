import { NextResponse } from 'next/server';

// Mock bookings storage (in production, use database)
const bookings: any[] = [];

export async function POST(request: Request) {
  const body = await request.json();
  const booking = {
    id: Math.random().toString(36).substr(2, 9),
    ...body,
    createdAt: new Date(),
  };
  bookings.push(booking);
  return NextResponse.json(booking, { status: 201 });
}

export async function GET() {
  return NextResponse.json(bookings);
}
