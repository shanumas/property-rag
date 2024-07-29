import "react-toastify/dist/ReactToastify.css";
import { Card, CardBody, Heading } from "@chakra-ui/react";
import { sendFeedback } from "../utils/sendFeedback";

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
  return (
    <Card
      onClick={async () => {
      }}
      cursor={"pointer"}
      alignSelf={"stretch"}
      overflow={"hidden"}
      justifyContent="center"
      alignItems="center"
      className="responsive-thumbnail"
    >
      <Heading size={"sm"} fontWeight={"normal"} color={"white"} padding={"5px"}>
        <img src={source} className="thumbnail" />
      </Heading>
    </Card>
  );
}
