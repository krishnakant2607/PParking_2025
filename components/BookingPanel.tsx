'use client';

import { useParkingStore } from '@/lib/store';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import { useState } from 'react';

export default function BookingPanel() {
  const { selectedGarage, selectedDate, setSelectedDate, predictions, setPredictions, addBooking } = useParkingStore();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showBookingModal, setShowBookingModal] = useState(false);
  const [paymentMethod, setPaymentMethod] = useState<'upi' | 'card'>('upi');

  const handleDateChange = (date: Date | null) => {
    if (date && date < new Date()) {
      setError('Cannot select a date in the past.');
      setSelectedDate(null);
    } else {
      setError(null);
      setSelectedDate(date);
    }
  };

  const handleCheckOccupancy = async () => {
    if (!selectedGarage || !selectedDate) return;
    
    setLoading(true);
    // Mock API call
    setTimeout(() => {
      const mockPrediction = Math.floor(Math.random() * 30) + 60; // Predict 60-90%
      setPredictions({ [selectedGarage.id]: mockPrediction });
      setLoading(false);
    }, 750);
  };

  const handleBook = async () => {
    if (!selectedGarage || !selectedDate) return;

    const newBooking = {
      id: `booking_${Date.now()}`,
      garageName: selectedGarage.name,
      date: selectedDate.toISOString(),
      amount: selectedGarage.price,
    };
    addBooking(newBooking);
    alert(`✅ Booking confirmed! ID: ${newBooking.id}`);
    setShowBookingModal(false);
  };

  const predictedOccupancy = predictions[selectedGarage?.id || ''] || null;

  return (
    <div className="space-y-4 pt-4">
      <h3 className="text-sm font-semibold text-gray-800">Pre-Book Your Spot</h3>
      
      {/* Date Picker */}
      <div>
        <DatePicker
          withPortal
          selected={selectedDate}
          onChange={handleDateChange}
          showTimeSelect
          filterTime={(time) => new Date(time) > new Date()}
          dateFormat="MMMM d, yyyy h:mm aa"
          minDate={new Date()}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 bg-gray-50"
          placeholderText="Choose parking date & time..."
        />
        {error && <p className="text-xs text-red-500 mt-1">{error}</p>}
        {selectedDate && !error && (
           <p className="text-xs text-green-600 mt-1">Date selected: {selectedDate.toLocaleString()}</p>
        )}
      </div>

      {/* Occupancy Prediction Display */}
      {predictedOccupancy !== null && (
        <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg text-center">
          <p className="text-sm font-bold text-blue-900">
            Predicted Occupancy: {predictedOccupancy}%
          </p>
          {predictedOccupancy > 80 && (
            <p className="text-xs text-orange-600 mt-1 font-semibold">High demand expected—book now!</p>
          )}
        </div>
      )}

      {/* Buttons */}
      <div className="space-y-3 pt-2">
        <button
          onClick={handleCheckOccupancy}
          disabled={loading || !selectedDate}
          className="w-full px-4 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition"
        >
          {loading ? 'Checking...' : 'Check Future Availability'}
        </button>
        <button
          onClick={() => setShowBookingModal(true)}
          disabled={!selectedDate}
          className="w-full px-4 py-3 bg-emerald-600 text-white font-semibold rounded-lg hover:bg-emerald-700 disabled:bg-gray-400 transition"
        >
          Pre-Book & Pay
        </button>
      </div>

      {/* Booking Modal */}
      {showBookingModal && selectedGarage && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-2xl max-w-sm w-full">
            <div className="p-6 text-center">
              <h3 className="text-lg font-bold text-gray-900">Confirm Booking</h3>
              <p className="text-sm text-gray-500 mt-1">Finalize your reservation for {selectedGarage.name}.</p>
            </div>

            <div className="px-6 pb-6 space-y-4">
              <div className="p-4 bg-gray-50 rounded-lg border border-gray-200 space-y-2">
                <div className="flex justify-between text-sm"><span className="text-gray-600">Date:</span> <span className="font-semibold">{selectedDate?.toLocaleDateString()}</span></div>
                <div className="flex justify-between text-sm"><span className="text-gray-600">Time:</span> <span className="font-semibold">{selectedDate?.toLocaleTimeString()}</span></div>
                <div className="flex justify-between text-lg font-bold mt-2 pt-2 border-t"><span className="text-gray-900">Total:</span> <span className="text-emerald-600">₹{selectedGarage.price}</span></div>
              </div>

              <div>
                <p className="text-sm font-semibold text-gray-700 mb-2">Pay via</p>
                <div className="flex gap-3">
                  <button onClick={() => setPaymentMethod('upi')} className={`flex-1 p-3 rounded-lg border-2 transition ${paymentMethod === 'upi' ? 'border-blue-500 bg-blue-50' : 'border-gray-300'}`}>UPI</button>
                  <button onClick={() => setPaymentMethod('card')} className={`flex-1 p-3 rounded-lg border-2 transition ${paymentMethod === 'card' ? 'border-blue-500 bg-blue-50' : 'border-gray-300'}`}>Card</button>
                </div>
              </div>
            </div>
            
            <div className="flex">
              <button onClick={() => setShowBookingModal(false)} className="w-1/2 p-4 text-sm font-semibold bg-gray-100 hover:bg-gray-200 rounded-bl-xl transition">Cancel</button>
              <button onClick={handleBook} className="w-1/2 p-4 text-sm font-semibold text-white bg-emerald-600 hover:bg-emerald-700 rounded-br-xl transition">Confirm & Pay</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
