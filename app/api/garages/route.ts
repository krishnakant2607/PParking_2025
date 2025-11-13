import { NextResponse } from 'next/server';

// Mock garage data
const mockGarages = [
  {
    id: '1',
    name: 'Downtown Garage',
    address: '123 Main St, City Center',
    price: 50,
    available: 8,
    total: 20,
    lat: 12.9716,
    lon: 77.5946,
  },
  {
    id: '2',
    name: 'Airport Parking',
    address: '456 Aviation Blvd',
    price: 40,
    available: 15,
    total: 40,
    lat: 13.1939,
    lon: 77.7068,
  },
  {
    id: '3',
    name: 'Mall Garage',
    address: '789 Shopping Center',
    price: 35,
    available: 5,
    total: 30,
    lat: 12.9352,
    lon: 77.6245,
  },
];

export async function GET() {
  return NextResponse.json(mockGarages);
}
