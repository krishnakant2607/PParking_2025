'use client';

import { useParkingStore, Garage } from '@/lib/store';
import { useState, useMemo } from 'react';

export default function GarageDropdown() {
  const { garages, setSelectedGarage, selectedGarage } = useParkingStore();
  const [searchTerm, setSearchTerm] = useState('');

  const filteredGarages = useMemo(() => {
    if (!searchTerm) {
      return garages;
    }
    return garages.filter((g: Garage) =>
      g.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      g.address.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [searchTerm, garages]);

  const handleSelect = (id: string) => {
    const garage = garages.find((g: Garage) => g.id === id);
    if (garage) {
      setSelectedGarage(garage);
    }
  };

  return (
    <div className="space-y-4">
      {/* Search Bar */}
      <div>
        <label className="block text-sm font-semibold text-gray-700 mb-2">Search Location</label>
        <input
          type="text"
          placeholder="Filter by name or address..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 bg-gray-50"
        />
      </div>

      {/* Dropdown */}
      <div>
        <label className="block text-sm font-semibold text-gray-700 mb-2">Select Garage</label>
        <select
          value={selectedGarage?.id || ''}
          onChange={(e) => handleSelect(e.target.value)}
          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 bg-gray-50 text-sm"
        >
          <option value="">Choose a parking garage...</option>
          {filteredGarages.map((g: Garage) => (
            <option key={g.id} value={g.id}>
              {g.name}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
