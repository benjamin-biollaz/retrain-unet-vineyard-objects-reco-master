from image_manager import ImageManager
from shapely.geometry import Polygon


class MetricManager:

    def compute_bounding_box_IOU(self, res_img, mask, class_color):
        
        # Extract list of bounding boxes for the prediction and mask
        image_manager = ImageManager(0, 0)
        mask_bbox_list = image_manager.extract_bounding_boxes_from_segmentation_mask(mask, class_color)
        img_bbox_list = image_manager.extract_bounding_boxes_from_segmentation_mask(res_img, class_color)

        sum_iou = 0

        # No intersection if the mask or the prediction has no bounding boxes
        if len(img_bbox_list) == 0 or len(mask_bbox_list) == 0:
            return 0

        for pred_bbox in img_bbox_list:
            pred_poly = Polygon(pred_bbox)

            # Compare the bounding box with each box of the ground truth
            iou_list = []
            for mask_bbox in mask_bbox_list:
                mask_poly = Polygon(mask_bbox)            
                union = pred_poly.union(mask_poly).area
                iou = 0

                # To avoid division by zero
                if (union != 0 and pred_poly.intersects(mask_poly)):
                    iou = pred_poly.intersection(mask_poly).area / pred_poly.union(mask_poly).area
                iou_list.append(iou)
            
            # Keep max value only
            sum_iou += max(iou_list)
        
        # Average IOU
        return  sum_iou / len(img_bbox_list)
        