import { create } from 'zustand';

// --- MOCK DATA ---
// This data simulates what would typically come from a backend API.
export const mockGarages = [
  {
    id: 'g1',
    name: 'Phoenix Marketcity',
    address: 'Whitefield Main Road, Mahadevapura',
    price: 75,
    available: 25,
    total: 250,
    lat: 12.9961,
    lon: 77.6963,
    forecast: [
      { time: '2 PM', occupancy: 60 },
      { time: '3 PM', occupancy: 75 },
      { time: '4 PM', occupancy: 85 },
      { time: '5 PM', occupancy: 90 },
      { time: '6 PM', occupancy: 80 },
    ],
  },
  {
    id: 'g2',
    name: 'UB City Parking',
    address: 'Vittal Mallya Road, KG Halli',
    price: 150,
    available: 50,
    total: 150,
    lat: 12.9719,
    lon: 77.5973,
    forecast: [
      { time: '2 PM', occupancy: 40 },
      { time: '3 PM', occupancy: 50 },
      { time: '4 PM', occupancy: 65 },
      { time: '5 PM', occupancy: 70 },
      { time: '6 PM', occupancy: 60 },
    ],
  },
  {
    id: 'g3',
    name: 'Bangalore International Airport',
    address: 'KIAL Rd, Devanahalli',
    price: 100,
    available: 800,
    total: 2000,
    lat: 13.1986,
    lon: 77.7066,
    forecast: [
      { time: '2 PM', occupancy: 70 },
      { time: '3 PM', occupancy: 72 },
      { time: '4 PM', occupancy: 75 },
      { time: '5 PM', occupancy: 80 },
      { time: '6 PM', occupancy: 82 },
    ],
  },
  {
    id: 'g4',
    name: 'Orion Mall',
    address: 'Dr. Rajkumar Road, Rajajinagar',
    price: 60,
    available: 120,
    total: 500,
    lat: 13.0112,
    lon: 77.5549,
    forecast: [
      { time: '2 PM', occupancy: 55 },
      { time: '3 PM', occupancy: 68 },
      { time: '4 PM', occupancy: 78 },
      { time: '5 PM', occupancy: 85 },
      { time: '6 PM', occupancy: 75 },
    ],
  },
];

// --- ZUSTAND STORE DEFINITION ---

export interface Garage {
  id: string;
  name: string;
  address: string;
  price: number;
  available: number;
  total: number;
  lat: number;
  lon: number;
  forecast: { time: string; occupancy: number }[];
}

export interface ParkingStore {
  garages: Garage[];
  selectedGarage: Garage | null;
  userLocation: { lat: number; lon: number } | null;
  selectedDate: Date | null;
  predictions: Record<string, number>;
  bookings: any[];
  setGarages: (garages: Garage[]) => void;
  setSelectedGarage: (garage: Garage | null) => void;
  setUserLocation: (location: { lat: number; lon: number } | null) => void;
  setSelectedDate: (date: Date | null) => void;
  setPredictions: (predictions: Record<string, number>) => void;
  addBooking: (booking: any) => void;
}

export const useParkingStore = create<ParkingStore>((set) => ({
  // Initialize state with mock data
  garages: mockGarages,
  selectedGarage: null,
  userLocation: { lat: 12.9716, lon: 77.5946 }, // Default to Bangalore center
  selectedDate: null,
  predictions: {},
  bookings: [],
  
  // Actions
  setGarages: (garages) => set({ garages }),
  setSelectedGarage: (garage) => set({ selectedGarage: garage, selectedDate: null, predictions: {} }),
  setUserLocation: (location) => set({ userLocation: location }),
  setSelectedDate: (date) => set({ selectedDate: date }),
  setPredictions: (predictions) => set({ predictions }),
  addBooking: (booking) =>
    set((state) => ({ bookings: [...state.bookings, booking] })),
}));