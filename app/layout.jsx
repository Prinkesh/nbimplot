import "./styles.css";

export const metadata = {
  title: "nbimplot Web Demo",
  description: "Standalone ImPlot and WASM plotting demo for browser apps.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
