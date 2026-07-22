import "./styles.css";

const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";

export const metadata = {
  title: "nbimplot Web Demo",
  description: "Standalone ImPlot and WASM plotting demo for browser apps.",
  icons: {
    icon: `${basePath}/favicon.svg`,
  },
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
