import React, { useState } from "react";

interface MRIOverlayProps {
  mriUrl: string;   // URL to the MRI image
  maskUrl: string;  // URL to the mask PNG
}

const MRIOverlay: React.FC<MRIOverlayProps> = ({ mriUrl, maskUrl }) => {
  const [mriLoaded, setMriLoaded] = useState(false);
  const [maskLoaded, setMaskLoaded] = useState(false);

  return (
    <div style={{ position: "relative", width: "512px", height: "512px" }}>
      {/* MRI base image */}
      <img
        src={mriUrl}
        alt="MRI"
        style={{
          width: "100%",
          height: "100%",
          display: mriLoaded ? "block" : "none",
        }}
        onLoad={() => setMriLoaded(true)}
      />
      {!mriLoaded && <div>Loading MRI...</div>}

      {/* Mask overlay */}
      <img
        src={maskUrl}
        alt="Mask"
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          opacity: 0.5, // Adjust for desired transparency
          pointerEvents: "none",
          mixBlendMode: "screen", // Optional: makes overlay blend nicely
          display: maskLoaded ? "block" : "none",
        }}
        onLoad={() => setMaskLoaded(true)}
      />
      {!maskLoaded && <div style={{ position: "absolute", top: 0, left: 0 }}>Loading mask...</div>}
    </div>
  );
};

export default MRIOverlay;