import "./styles.css";

const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";
const siteUrl = "https://prinkesh.github.io/nbimplot/";
const githubUrl = "https://github.com/Prinkesh/nbimplot";
const pypiUrl = "https://pypi.org/project/nbimplot/";
const npmUrl = "https://www.npmjs.com/package/@nbimplot/web";

export const metadata = {
  metadataBase: new URL(siteUrl),
  title: "nbimplot - Fast ImPlot WASM Plotting For Notebooks And Web Apps",
  description:
    "ImPlot-powered WASM plotting for Jupyter notebooks and browser apps, engineered for interactive million-point plots with binary transport and screen-resolution LOD.",
  keywords: [
    "fast Jupyter plotting",
    "million point plotting",
    "ImPlot Jupyter",
    "WASM plotting",
    "WebGL2 plotting",
    "large data visualization",
    "interactive time series",
    "notebook plotting",
    "typed array plotting",
  ],
  alternates: {
    canonical: siteUrl,
  },
  openGraph: {
    title: "nbimplot - Fast ImPlot WASM Plotting",
    description:
      "Strict WASM + ImGui + ImPlot plotting for notebooks and web apps, with binary data transport and LOD for large line plots.",
    url: siteUrl,
    siteName: "nbimplot",
    type: "website",
  },
  twitter: {
    card: "summary",
    title: "nbimplot - Fast ImPlot WASM Plotting",
    description:
      "ImPlot-quality interactivity in notebooks and web apps, engineered for million-point plots.",
  },
  icons: {
    icon: `${basePath}/favicon.svg`,
  },
};

const structuredData = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "SoftwareApplication",
      "@id": `${siteUrl}#software`,
      name: "nbimplot",
      applicationCategory: "DeveloperApplication",
      applicationSubCategory: "Data visualization",
      operatingSystem: "Web browser, Jupyter",
      description:
        "ImPlot-powered WASM plotting for Jupyter notebooks and browser web apps, optimized for interactive large numeric arrays and million-point line plots.",
      url: siteUrl,
      codeRepository: githubUrl,
      downloadUrl: pypiUrl,
      installUrl: pypiUrl,
      softwareVersion: "0.1.10",
      programmingLanguage: ["Python", "JavaScript", "C++", "WebAssembly"],
      runtimePlatform: ["Jupyter", "WebGL2", "WebAssembly"],
      keywords:
        "Jupyter plotting, ImPlot, ImGui, WASM plotting, WebGL2, million point plotting, large data visualization, time series plotting",
      license: "https://github.com/Prinkesh/nbimplot/blob/main/LICENSE",
      sameAs: [githubUrl, pypiUrl, npmUrl],
      offers: {
        "@type": "Offer",
        price: "0",
        priceCurrency: "USD",
      },
    },
    {
      "@type": "SoftwareSourceCode",
      "@id": `${siteUrl}#source`,
      name: "nbimplot source code",
      codeRepository: githubUrl,
      programmingLanguage: ["Python", "JavaScript", "C++"],
      runtimePlatform: ["Jupyter", "WebAssembly", "WebGL2"],
      license: "https://github.com/Prinkesh/nbimplot/blob/main/LICENSE",
      targetProduct: {
        "@id": `${siteUrl}#software`,
      },
    },
  ],
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }}
        />
        {children}
      </body>
    </html>
  );
}
