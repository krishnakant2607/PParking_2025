import type { Metadata } from 'next';
import './globals.css';
import Header from '@/components/Header';

export const metadata: Metadata = {
  title: 'IntelliPark - Smart Parking Finder',
  description: 'Find and book parking spaces intelligently',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="h-screen flex flex-col bg-gray-100">
        <Header />
        {children}
      </body>
    </html>
  );
}
