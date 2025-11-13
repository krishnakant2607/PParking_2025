'use client';

import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts';

interface OccupancyChartProps {
  data?: Array<{ time: string; occupancy: number }>;
}

export default function OccupancyForecast({ data }: OccupancyChartProps) {
  const mockData = data || [
    { time: '9 AM', occupancy: 35 },
    { time: '10 AM', occupancy: 48 },
    { time: '11 AM', occupancy: 62 },
    { time: '12 PM', occupancy: 78 },
    { time: '1 PM', occupancy: 85 },
    { time: '2 PM', occupancy: 72 },
    { time: '3 PM', occupancy: 55 },
    { time: '4 PM', occupancy: 42 },
    { time: '5 PM', occupancy: 68 },
  ];

  return (
    <div className="bg-white shadow-md p-6 rounded-lg">
      <h2 className="text-lg font-semibold mb-4">📈 Occupancy Forecast (24h)</h2>
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={mockData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" />
          <YAxis label={{ value: 'Occupancy %', angle: -90, position: 'insideLeft' }} />
          <Tooltip 
            contentStyle={{ backgroundColor: '#f0f0f0', borderRadius: '8px' }}
            formatter={(value) => `${value}%`}
          />
          <Line 
            type="monotone" 
            dataKey="occupancy" 
            stroke="#2563eb" 
            strokeWidth={2}
            dot={{ fill: '#2563eb', r: 4 }}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
