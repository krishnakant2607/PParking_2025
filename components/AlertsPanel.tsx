export default function AlertsPanel() {
  return (
    <div className="p-4 rounded-xl bg-gradient-to-br from-purple-50 to-indigo-100 border border-purple-200">
      <div className="flex justify-between items-center">
        <div>
          <h3 className="font-bold text-purple-900">Smart Alerts</h3>
          <p className="text-xs text-purple-700 mt-1">Get notified when spots open up.</p>
        </div>
        <label className="relative inline-flex items-center cursor-pointer">
          <input type="checkbox" value="" className="sr-only peer" />
          <div className="w-11 h-6 bg-gray-300 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
        </label>
      </div>
    </div>
  );
}
