'use client';

import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend } from 'recharts';

interface RevenueChartProps {
  data?: Array<{ day: string; revenue: number; predicted: number }>;
}

export default function RevenueGraph({ data }: RevenueChartProps) {
  const mockData = data || [
    { day: 'Mon', revenue: 4200, predicted: 4500 },
    { day: 'Tue', revenue: 3800, predicted: 4100 },
    { day: 'Wed', revenue: 5100, predicted: 5200 },
    { day: 'Thu', revenue: 4900, predicted: 5000 },
    { day: 'Fri', revenue: 6200, predicted: 6300 },
    { day: 'Sat', revenue: 7100, predicted: 7200 },
    { day: 'Sun', revenue: 5800, predicted: 6000 },
  ];

  return (
    <div className="bg-white shadow-md p-6 rounded-lg">
      <h2 className="text-lg font-semibold mb-4">💰 Revenue (7 days + Forecast)</h2>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={mockData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="day" />
          <YAxis label={{ value: 'Revenue ₹', angle: -90, position: 'insideLeft' }} />
          <Tooltip 
            contentStyle={{ backgroundColor: '#f0f0f0', borderRadius: '8px' }}
            formatter={(value) => `₹${value}`}
          />
          <Legend />
          <Bar dataKey="revenue" fill="#10b981" name="Actual Revenue" />
          <Bar dataKey="predicted" fill="#a3e635" name="Forecast" opacity={0.7} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
