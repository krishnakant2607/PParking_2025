import { NextResponse } from 'next/server';

// Mock prediction data
export async function POST(request: Request) {
  const body = await request.json();
  const { garage_id, datetime } = body;

  // Simple mock: return occupancy based on time of day
  const hour = new Date(datetime).getHours();
  const baseOccupancy = 50;
  const occupancy = Math.min(
    95,
    baseOccupancy + (Math.sin((hour - 9) * Math.PI / 12) * 40)
  );

  return NextResponse.json({
    garage_id,
    datetime,
    predicted_occupancy: Math.round(occupancy),
    confidence: 0.85,
  });
}
