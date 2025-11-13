'use client';

import dynamic from 'next/dynamic';
import GarageDropdown from '@/components/GarageDropdown';
import FeaturePanel from '@/components/FeaturePanel';
import { useParkingStore } from '@/lib/store';

// Dynamically import map to avoid SSR issues with Leaflet
const ParkingMap = dynamic(() => import('@/components/ParkingMap'), {
  ssr: false,
  loading: () => <div className="w-full h-full bg-gray-200 flex items-center justify-center"><p className="text-gray-500">Loading Map...</p></div>,
});

export default function Home() {
  const { selectedGarage } = useParkingStore();

  return (
    <main className="flex flex-grow">
      {/* Left Panel: Controls & Details */}
      <div className="w-[40%] h-full bg-white border-r border-gray-200 overflow-y-auto">
        <div className="p-6 space-y-6">
          {/* Header */}
          <div className="pb-4 border-b border-gray-200">
            <h1 className="text-2xl font-bold text-gray-900">Find Your Spot</h1>
            <p className="text-sm text-gray-500">Select a garage or search to see details</p>
          </div>
          
          {/* Garage Selection */}
          <GarageDropdown />

          {/* Conditional Details */}
          {selectedGarage ? (
            <FeaturePanel />
          ) : (
            <div className="p-8 text-center text-gray-400 bg-gray-50 rounded-xl">
              <p className="font-medium">Welcome to IntelliPark</p>
              <p className="text-sm">Your smart parking solution.</p>
            </div>
          )}
        </div>
      </div>

      {/* Right Panel: Map */}
      <div className="w-[60%] h-full relative z-0">
        <ParkingMap />
      </div>
    </main>
  );
}
