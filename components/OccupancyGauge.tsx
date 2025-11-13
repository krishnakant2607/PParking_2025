'use client';

interface OccupancyGaugeProps {
  percentage?: number;
}

export default function OccupancyGauge({ percentage = 72 }: OccupancyGaugeProps) {
  const circumference = 2 * Math.PI * 45;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;
  
  const getColor = (percent: number) => {
    if (percent < 50) return '#10b981';
    if (percent < 80) return '#f59e0b';
    return '#ef4444';
  };

  return (
    <div className="bg-white border border-gray-200 p-8 rounded-xl flex flex-col items-center shadow-sm hover:shadow-md transition">
      <h2 className="text-lg font-bold text-gray-900 mb-6">Current Occupancy</h2>
      
      <div className="relative w-56 h-56">
        <svg width="100%" height="100%" viewBox="0 0 120 120">
          {/* Background circle */}
          <circle
            cx="60"
            cy="60"
            r="45"
            fill="none"
            stroke="#e5e7eb"
            strokeWidth="8"
          />
          
          {/* Progress circle */}
          <circle
            cx="60"
            cy="60"
            r="45"
            fill="none"
            stroke={getColor(percentage)}
            strokeWidth="8"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            style={{ transition: 'stroke-dashoffset 0.5s ease' }}
            transform="rotate(-90 60 60)"
          />
        </svg>
        
        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <div className="text-5xl font-bold text-gray-900">{percentage}%</div>
          <div className="text-xs text-gray-500 uppercase tracking-wide mt-1">Occupied</div>
        </div>
      </div>

      {/* Status Bar */}
      <div className="mt-8 w-full px-2">
        <div className="flex justify-between text-xs font-semibold text-gray-600 mb-2">
          <span>Available</span>
          <span>Full</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
          <div 
            className={`h-full transition-all duration-500 ${
              percentage < 50 ? 'bg-emerald-500' :
              percentage < 80 ? 'bg-amber-500' : 'bg-red-500'
            }`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>

      {/* Status Text */}
      <div className="mt-6 text-center">
        <div className="inline-block px-4 py-2 rounded-full" style={{
          backgroundColor: percentage < 50 ? '#d1fae5' : percentage < 80 ? '#fef3c7' : '#fee2e2',
        }}>
          <p className="text-sm font-semibold" style={{
            color: percentage < 50 ? '#047857' : percentage < 80 ? '#92400e' : '#991b1b',
          }}>
            {percentage < 50 && 'Plenty of spaces available'}
            {percentage >= 50 && percentage < 80 && 'Occupancy moderate'}
            {percentage >= 80 && 'Approaching capacity'}
          </p>
        </div>
      </div>
    </div>
  );
}
