'use client';

import InfoPanel from './InfoPanel';
import BookingPanel from './BookingPanel';
import AlertsPanel from './AlertsPanel';
import WalkInPanel from './WalkInPanel';

export default function FeaturePanel() {
  return (
    <div className="space-y-6">
      <InfoPanel />
      <hr className="border-gray-200" />
      <BookingPanel />
      <hr className="border-gray-200" />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <AlertsPanel />
        <WalkInPanel />
      </div>
    </div>
  );
}
