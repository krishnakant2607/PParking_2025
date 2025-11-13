import Link from 'next/link';

const Header = () => {
  return (
    <header className="bg-gray-800 text-white p-4">
      <div className="container mx-auto flex justify-between items-center">
        <Link href="/" className="text-2xl font-bold">
          IntelliPark
        </Link>
        <nav>
          <Link href="/" className="mr-4 hover:text-gray-300">
            User Home
          </Link>
          <Link href="/dashboard" className="hover:text-gray-300">
            Admin Dashboard
          </Link>
        </nav>
      </div>
    </header>
  );
};

export default Header;
