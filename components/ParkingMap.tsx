'use client';

import { useParkingStore, Garage } from '@/lib/store';
import { useEffect, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix Leaflet default icons
const defaultIcon = L.icon({
  iconUrl:
    'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  iconRetinaUrl:
    'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  shadowUrl:
    'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

export default function ParkingMap() {
  const { garages, selectedGarage } = useParkingStore();
  const [mapInstance, setMapInstance] = useState<L.Map | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted || !mapInstance) return;

    // Initialize map only once
    if (mapInstance) {
      // Add markers
      garages.forEach((garage: Garage) => {
        L.marker([garage.lat, garage.lon], { icon: defaultIcon })
          .bindPopup(`<div class="text-sm"><p class="font-semibold">${garage.name}</p><p class="text-xs">Available: ${garage.available}/${garage.total}</p><p class="text-xs">₹${garage.price}/hr</p></div>`)
          .addTo(mapInstance);
      });
    }
  }, [mapInstance, mounted, garages]);

  useEffect(() => {
    if (selectedGarage && mapInstance) {
      mapInstance.flyTo([selectedGarage.lat, selectedGarage.lon], 15, {
        duration: 1,
      });
    }
  }, [selectedGarage, mapInstance]);

  useEffect(() => {
    if (!mounted) return;

    // Clear existing map instance if it exists
    const existingContainer = document.getElementById('map');
    if (existingContainer && (existingContainer as any)._leaflet_map) {
      (existingContainer as any)._leaflet_map.remove();
    }

    const map = L.map('map', {
      center: [12.9716, 77.5946],
      zoom: 13,
    });

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors',
      maxZoom: 19,
    }).addTo(map);

    setMapInstance(map);

    return () => {
      // Cleanup on unmount
      if (map) {
        map.remove();
      }
    };
  }, [mounted]);

  if (!mounted) {
    return (
      <div className="w-full h-full bg-gray-200 flex items-center justify-center text-gray-600">
        🗺️ Loading map...
      </div>
    );
  }

  return <div id="map" className="w-full h-full" />;
}
