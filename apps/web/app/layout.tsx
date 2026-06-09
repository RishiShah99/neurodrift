import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NeuroDrift — brain trajectory playground",
  description:
    "Upload a T1 MRI. Drag age. Toggle ApoE4. Pick a treatment. Rotate the brain. In a browser.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="font-sans antialiased min-h-screen">{children}</body>
    </html>
  );
}
