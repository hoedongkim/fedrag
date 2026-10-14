import { shareImage } from "@/components/ShareImage";
import { SHARE_IMAGE_ALT, SHARE_IMAGE_SIZE } from "@/lib/site";

export const alt = SHARE_IMAGE_ALT;
export const size = SHARE_IMAGE_SIZE;
export const contentType = "image/png";

export default function TwitterImage() {
  return shareImage();
}
