# IntelliPark

A smart parking management system built with Next.js 15, featuring real-time occupancy tracking, predictive analytics, and dynamic pricing.

## Live Demo

🔗 **[View Live Application](https://intellipark-git-main-krishnakant2607s-projects.vercel.app/)**

https://intellipark-git-main-krishnakant2607s-projects.vercel.app/

## Features

- **Real-time Parking Search** - Interactive map with garage locations
- **Occupancy Prediction** - AI-powered parking availability forecasts
- **Online Booking** - Reserve parking spots with integrated payment options
- **Operator Dashboard** - Analytics, dynamic pricing, and booking management
- **Responsive Design** - Mobile-friendly interface

## Tech Stack

- **Framework:** Next.js 15 (App Router)
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **State Management:** Zustand
- **Maps:** Leaflet.js with OpenStreetMap
- **Charts:** Recharts
- **Deployment:** Vercel

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

```bash
# Clone the repository
git clone git@github.com:krishnakant2607/PParking_2025.git

# Navigate to project directory
cd PParking_2025

# Install dependencies
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
├── app/
│   ├── api/          # API routes (garages, predictions, bookings)
│   ├── dashboard/    # Operator dashboard
│   └── page.tsx      # Main landing page
├── components/       # React components
├── lib/             # State management and utilities
└── public/          # Static assets
```

## Available Routes

- `/` - User dashboard with parking search and map
- `/dashboard` - Operator control panel

## API Endpoints

- `GET /api/garages` - Fetch available parking garages
- `POST /api/predict` - Get occupancy predictions
- `POST /api/bookings` - Create new booking

## Build for Production

```bash
npm run build
npm start
```

## Contact

GitHub: [@krishnakant2607](https://github.com/krishnakant2607)
