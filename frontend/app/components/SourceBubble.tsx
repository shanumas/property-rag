import React, { useState, useEffect } from 'react';
import "react-toastify/dist/ReactToastify.css";
import Zoom from 'react-medium-image-zoom';
import 'react-medium-image-zoom/dist/styles.css';
import { Card, Heading } from "@chakra-ui/react";

export type Source = {
  url: string;
  images: string;
};

export function SourceBubble({
  source,
  runId,
}: {
  source: string;
  runId?: string;
}) {
  const [isFullScreen, setIsFullScreen] = useState(false);

  useEffect(() => {
    const handleKeyUp = (event:any) => {
      if (event.key === 'Escape') {
        setIsFullScreen(false);
      }
    };

    window.addEventListener('keyup', handleKeyUp);
    return () => window.removeEventListener('keyup', handleKeyUp);
  }, []);

  const handleImageClick = () => {
    setIsFullScreen(true);
  };

  const handleClose = () => {
    setIsFullScreen(false);
  };

  return (
    <>
      <Card
        onClick={handleImageClick}
        cursor={"pointer"}
        alignSelf={"stretch"}
        overflow={"hidden"}
        justifyContent="center"
        alignItems="center"
        className="responsive-thumbnail"
      >
        <Heading size={"sm"} fontWeight={"normal"} color={"white"} padding={"5px"}>
          <img src={source} className="thumbnail" alt="Thumbnail" />
        </Heading>
      </Card>

      {isFullScreen && (
        <div className="fullscreen-overlay" onClick={handleClose}>
          <div className="fullscreen-overlay" onClick={handleClose}>
          <div className="fullscreen-container">
            <Zoom>
              <img src={source} className="fullscreen-image" alt="Full Screen" />
            </Zoom>
            <button className="close-button" onClick={handleClose}>X</button>
          </div>
        </div>
        </div>
      )}
    </>
  );
}
