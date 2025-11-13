'use client';

import { useParkingStore } from '@/lib/store';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

export default function InfoPanel() {
  const { selectedGarage } = useParkingStore();

  if (!selectedGarage) return null;

  const occupancyPercentage = Math.round(
    (selectedGarage.available / selectedGarage.total) * 100
  );

  return (
    <div className="space-y-6">
      {/* Name & Address */}
      <div>
        <h2 className="text-3xl font-bold text-gray-900">{selectedGarage.name}</h2>
        <p className="text-sm text-gray-500 mt-1">{selectedGarage.address}</p>
      </div>

      {/* Info Grid */}
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg border border-blue-200">
          <p className="text-xs font-semibold text-blue-600 uppercase tracking-wide mb-1">Dynamic Price</p>
          <p className="text-3xl font-bold text-blue-900">₹{selectedGarage.price}<span className="text-sm text-blue-700">/hr</span></p>
        </div>
        <div className="p-4 bg-gradient-to-br from-emerald-50 to-emerald-100 rounded-lg border border-emerald-200">
          <p className="text-xs font-semibold text-emerald-600 uppercase tracking-wide mb-1">Live Availability</p>
          <p className="text-3xl font-bold text-emerald-900">{selectedGarage.available}<span className="text-sm text-emerald-700">/{selectedGarage.total}</span></p>
        </div>
      </div>

      {/* Occupancy Bar */}
      <div>
        <div className="flex justify-between items-center mb-1">
          <p className="text-xs font-semibold text-gray-600 uppercase tracking-wide">Current Occupancy</p>
          <p className="text-sm font-bold text-gray-900">{100 - occupancyPercentage}% Full</p>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2.5 overflow-hidden">
          <div
            className={`h-full transition-all duration-500 ease-out ${
              (100 - occupancyPercentage) > 80
                ? 'bg-red-500'
                : (100 - occupancyPercentage) > 50
                ? 'bg-yellow-400'
                : 'bg-green-500'
            }`}
            style={{ width: `${100 - occupancyPercentage}%` }}
          />
        </div>
      </div>
      
      {/* Occupancy Forecast */}
      <div className="pt-2">
         <h3 className="text-sm font-semibold text-gray-800 mb-3">Occupancy Forecast</h3>
        <div className="h-40 w-full">
           <ResponsiveContainer width="100%" height="100%">
            <LineChart data={selectedGarage.forecast} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis dataKey="time" stroke="#6b7280" fontSize={12} />
              <YAxis stroke="#6b7280" fontSize={12} domain={[0, 100]} unit="%" />
              <Tooltip
                contentStyle={{
                  borderRadius: '8px',
                  borderColor: '#d1d5db',
                  fontSize: '12px',
                  fontWeight: '600',
                }}
                labelStyle={{ color: '#1f2937' }}
                formatter={(value) => [`${value}% Full`, 'Occupancy']}
              />
              <Line type="monotone" dataKey="occupancy" stroke="#2563eb" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
