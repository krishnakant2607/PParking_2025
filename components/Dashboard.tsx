'use client';

import OccupancyGauge from '@/components/OccupancyGauge';
import OccupancyForecast from '@/components/OccupancyForecast';
import RevenueGraph from '@/components/RevenueGraph';
import { useState } from 'react';

export default function Dashboard() {
  const [dynamicPrice, setDynamicPrice] = useState(50);
  const [priceMultiplier, setPriceMultiplier] = useState(1.0);

  const handlePriceChange = (value: number) => {
    const newMultiplier = value / 50;
    setPriceMultiplier(newMultiplier);
    setDynamicPrice(value);
  };

  return (
    <div className="p-10 bg-gray-50">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-10">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Operator Dashboard</h1>
          <p className="text-gray-600 text-lg">Monitor, predict, and optimize your parking operations in real-time</p>
        </div>

        {/* Key Metrics Row */}
        <div className="grid grid-cols-4 gap-5 mb-10">
          <MetricCard
            title="Current Occupancy"
            value="72%"
            trend="+5%"
            trendType="up"
            color="blue"
          />
          <MetricCard
            title="Revenue (Today)"
            value="₹8,420"
            trend="+₹1,200"
            trendType="up"
            color="emerald"
          />
          <MetricCard
            title="Active Bookings"
            value="24"
            trend="+3"
            trendType="up"
            color="purple"
          />
          <MetricCard
            title="System Health"
            value="98%"
            trend="Optimal"
            trendType="neutral"
            color="slate"
          />
        </div>

        {/* Charts & Controls Row */}
        <div className="grid grid-cols-3 gap-6 mb-10">
          {/* Occupancy Gauge */}
          <OccupancyGauge percentage={72} />

          {/* Dynamic Pricing Control */}
          <div className="bg-white border border-gray-200 p-7 rounded-xl shadow-sm hover:shadow-md transition">
            <h2 className="text-lg font-bold text-gray-900 mb-6">Dynamic Pricing</h2>
            <div className="space-y-5">
              <div className="bg-gradient-to-br from-emerald-50 to-emerald-100 p-4 rounded-lg border border-emerald-200">
                <p className="text-xs font-semibold text-emerald-600 uppercase tracking-wide mb-1">Current Rate</p>
                <p className="text-3xl font-bold text-emerald-900">₹{dynamicPrice}<span className="text-sm text-emerald-700">/hr</span></p>
              </div>
              
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm font-semibold text-gray-900">Price Multiplier</label>
                  <span className="text-sm font-bold text-blue-600">{priceMultiplier.toFixed(2)}x</span>
                </div>
                <input
                  type="range"
                  min="25"
                  max="100"
                  value={dynamicPrice}
                  onChange={(e) => handlePriceChange(Number(e.target.value))}
                  className="w-full h-2 bg-gray-300 rounded-full appearance-none cursor-pointer accent-blue-600"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-2 font-medium">
                  <span>-50%</span>
                  <span>Normal</span>
                  <span>+100%</span>
                </div>
              </div>

              <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="font-semibold text-blue-900 text-sm mb-1">Smart Recommendation</p>
                <p className="text-blue-700 text-xs">
                  High demand forecasted for 5-6 PM. Consider increasing rate by 15%.
                </p>
              </div>

              <button className="w-full bg-blue-600 text-white px-4 py-3 rounded-lg font-semibold hover:bg-blue-700 transition transform hover:scale-105 active:scale-95">
                Apply Changes
              </button>
            </div>
          </div>

          {/* Alerts Panel */}
          <div className="bg-white border border-gray-200 p-7 rounded-xl shadow-sm hover:shadow-md transition">
            <h2 className="text-lg font-bold text-gray-900 mb-6">Alerts & Notifications</h2>
            <div className="space-y-3">
              <div className="p-4 bg-red-50 border-l-4 border-red-500 rounded-lg">
                <p className="text-sm font-semibold text-red-900">Camera #3 Offline</p>
                <p className="text-xs text-red-700 mt-1">Disconnected 5 minutes ago</p>
              </div>
              <div className="p-4 bg-amber-50 border-l-4 border-amber-500 rounded-lg">
                <p className="text-sm font-semibold text-amber-900">Occupancy Peak Expected</p>
                <p className="text-xs text-amber-700 mt-1">Forecast: 90% occupancy by 5 PM</p>
              </div>
              <div className="p-4 bg-emerald-50 border-l-4 border-emerald-500 rounded-lg">
                <p className="text-sm font-semibold text-emerald-900">All Systems Operational</p>
                <p className="text-xs text-emerald-700 mt-1">Last synced 2 minutes ago</p>
              </div>
            </div>
          </div>
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-2 gap-6 mb-10">
          <OccupancyForecast />
          <RevenueGraph />
        </div>

        {/* Booking Table */}
        <div className="bg-white border border-gray-200 p-7 rounded-xl shadow-sm">
          <h2 className="text-lg font-bold text-gray-900 mb-6">Active & Upcoming Bookings</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-gray-50">
                  <th className="text-left px-4 py-4 font-semibold text-gray-900">Customer</th>
                  <th className="text-left px-4 py-4 font-semibold text-gray-900">Parking Slot</th>
                  <th className="text-left px-4 py-4 font-semibold text-gray-900">Start Time</th>
                  <th className="text-left px-4 py-4 font-semibold text-gray-900">End Time</th>
                  <th className="text-left px-4 py-4 font-semibold text-gray-900">Status</th>
                  <th className="text-left px-4 py-4 font-semibold text-gray-900">Amount</th>
                  <th className="text-left px-4 py-4 font-semibold text-gray-900">Action</th>
                </tr>
              </thead>
              <tbody>
                <BookingRow
                  user="John Doe"
                  slot="A-101"
                  from="10:00 AM"
                  to="1:00 PM"
                  status="Active"
                  amount="₹150"
                />
                <BookingRow
                  user="Jane Smith"
                  slot="B-205"
                  from="2:00 PM"
                  to="5:00 PM"
                  status="Pending"
                  amount="₹200"
                />
                <BookingRow
                  user="Mike Johnson"
                  slot="C-312"
                  from="Tomorrow 9:00 AM"
                  to="Tomorrow 12:00 PM"
                  status="Pre-booked"
                  amount="₹175"
                />
                <BookingRow
                  user="Sarah Lee"
                  slot="A-205"
                  from="11:00 AM"
                  to="2:00 PM"
                  status="Completed"
                  amount="₹150"
                />
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({
  title,
  value,
  trend,
  trendType,
  color,
}: {
  title: string;
  value: string;
  trend: string;
  trendType: 'up' | 'down' | 'neutral';
  color: string;
}) {
  const colorMap: { [key: string]: { bg: string; text: string; gradient: string } } = {
    blue: { bg: 'bg-blue-50', text: 'text-blue-900', gradient: 'from-blue-50 to-blue-100' },
    emerald: { bg: 'bg-emerald-50', text: 'text-emerald-900', gradient: 'from-emerald-50 to-emerald-100' },
    purple: { bg: 'bg-purple-50', text: 'text-purple-900', gradient: 'from-purple-50 to-purple-100' },
    slate: { bg: 'bg-slate-50', text: 'text-slate-900', gradient: 'from-slate-50 to-slate-100' },
  };

  const selected = colorMap[color] || colorMap.blue;

  return (
    <div className={`bg-white border border-gray-200 p-6 rounded-xl shadow-sm hover:shadow-md transition bg-gradient-to-br ${selected.gradient}`}>
      <p className="text-gray-600 text-sm font-medium mb-3">{title}</p>
      <p className={`text-3xl font-bold ${selected.text} mb-3`}>{value}</p>
      <p
        className={`text-sm font-semibold flex items-center gap-1 ${
          trendType === 'up'
            ? 'text-emerald-600'
            : trendType === 'down'
            ? 'text-red-600'
            : 'text-slate-600'
        }`}
      >
        {trendType === 'up' ? '↑' : trendType === 'down' ? '↓' : '→'} {trend}
      </p>
    </div>
  );
}

function BookingRow({
  user,
  slot,
  from,
  to,
  status,
  amount,
}: {
  user: string;
  slot: string;
  from: string;
  to: string;
  status: string;
  amount: string;
}) {
  const statusStyles = {
    Active: 'bg-emerald-100 text-emerald-800 border border-emerald-300',
    Pending: 'bg-blue-100 text-blue-800 border border-blue-300',
    'Pre-booked': 'bg-purple-100 text-purple-800 border border-purple-300',
    Completed: 'bg-gray-100 text-gray-800 border border-gray-300',
  };

  return (
    <tr className="border-b hover:bg-gray-50 transition">
      <td className="px-4 py-4 font-semibold text-gray-900">{user}</td>
      <td className="px-4 py-4 text-gray-700">{slot}</td>
      <td className="px-4 py-4 text-gray-700">{from}</td>
      <td className="px-4 py-4 text-gray-700">{to}</td>
      <td className="px-4 py-4">
        <span
          className={`px-3 py-1 rounded-full text-xs font-semibold ${
            statusStyles[status as keyof typeof statusStyles]
          }`}
        >
          {status}
        </span>
      </td>
      <td className="px-4 py-4 font-bold text-gray-900">{amount}</td>
      <td className="px-4 py-4">
        <button className="text-blue-600 hover:text-blue-800 font-semibold text-sm transition hover:underline">
          {status === 'Active' ? 'Extend' : 'View Details'}
        </button>
      </td>
    </tr>
  );
}
