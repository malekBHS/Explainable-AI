import argparse
import math
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

class RFBoxCAM:
    def __init__(self, weights_path, layer_index=17, device=None):
        """
        Initializes the RF-BoxCAM explainer.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(weights_path).model.to(self.device)
        self.layer_index = layer_index

    @staticmethod
    def iou(box1, box2):
        """
        Calculate IoU between two bounding boxes.
        Boxes represented as (x_center, y_center, width, height).
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
        x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2

        x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
        x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_w = max(0, inter_xmax - inter_xmin)
        inter_h = max(0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        union_area = (w1 * h1) + (w2 * h2) - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def get_overlapping_boxes(self, results, target_index, conf_thresh=0.25, iou_thresh=0.45):
        """
        Finds and weights overlapping boxes targeting the same object using Softmax.
        """
        xt, yt, wt, ht = results[0:4, target_index]
        target_box = [xt, yt, wt, ht]
        
        valid_boxes = {}
        for i in range(results.shape[1]):
            conf = results[5, i]
            if conf < conf_thresh:
                continue
                
            current_box = results[0:4, i]
            if self.iou(current_box, target_box) > iou_thresh:
                valid_boxes[i] = conf

        # Softmax aggregation
        conf_sum = sum(math.exp(c) for c in valid_boxes.values())
        return {idx: math.exp(conf) / conf_sum for idx, conf in valid_boxes.items()}

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor_img = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return img_resized, tensor_img.to(self.device)

    def generate(self, image_path, target_index):
        """
        Generates the aggregated RF-BoxCAM explanation.
        """
        img_rgb, tensor_img = self.preprocess_image(image_path)
        tensor_img.requires_grad_(True)

        # Get raw predictions to identify the pre-NMS cluster
        with torch.no_grad():
            raw_preds = self.model(tensor_img)[0][0].cpu().numpy()
        
        box_weights = self.get_overlapping_boxes(raw_preds, target_index)
        final_heatmap = np.zeros((640, 640))

        # Hook storage
        activations = {}
        gradients = {}

        def forward_hook(module, input, output):
            activations['value'] = output

        def backward_hook(module, grad_inp, grad_out):
            gradients['value'] = grad_out[0]

        target_layer = self.model.model[self.layer_index]
        target_layer.requires_grad_(True)
        
        # Attach hooks dynamically and clean them up to prevent memory leaks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        # Single forward pass
        output = self.model(tensor_img)

        for box_idx, weight in box_weights.items():
            self.model.zero_grad(set_to_none=True)
            if tensor_img.grad is not None:
                tensor_img.grad.zero_()

            # Backward pass for the specific detection confidence
            score = output[0][0][5][box_idx]
            score.backward(retain_graph=True)

            act = activations['value']
            grad = gradients['value']

            # Compute active grid cells
            weighted_acts = grad[0] * act
            hm = torch.sum(weighted_acts[0], dim=0)
            hm = torch.relu(hm)

            non_zero_indices = torch.nonzero(hm).cpu().numpy()
            
            box_heatmap = np.zeros((640, 640))
            
            # Reconstruct exact input-level receptive fields
            for id_y, id_x in non_zero_indices:
                self.model.zero_grad(set_to_none=True)
                if tensor_img.grad is not None:
                    tensor_img.grad.zero_()
                    
                hm[id_y, id_x].backward(retain_graph=True)
                
                input_grad = tensor_img.grad.detach().cpu().numpy()[0]
                input_grad = np.transpose(input_grad, (1, 2, 0))
                gradmap = np.linalg.norm(input_grad, axis=2)
                
                # Smooth and normalize
                smoothed_rf = cv2.GaussianBlur(gradmap, (21, 21), 0)
                if np.max(smoothed_rf) > np.min(smoothed_rf):
                    smoothed_rf = (smoothed_rf - np.min(smoothed_rf)) / (np.max(smoothed_rf) - np.min(smoothed_rf))
                
                box_heatmap += hm[id_y, id_x].item() * smoothed_rf

            # Normalize box heatmap
            if np.max(box_heatmap) > np.min(box_heatmap):
                box_heatmap = (box_heatmap - np.min(box_heatmap)) / (np.max(box_heatmap) - np.min(box_heatmap))

            final_heatmap += weight * box_heatmap

        # Clean up hooks
        forward_handle.remove()
        backward_handle.remove()

        # Final normalization
        if np.max(final_heatmap) > np.min(final_heatmap):
            final_heatmap = (final_heatmap - np.min(final_heatmap)) / (np.max(final_heatmap) - np.min(final_heatmap))

        return img_rgb, final_heatmap

    def save_visualization(self, image, heatmap, save_path):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.imshow(heatmap, cmap='jet', alpha=0.4)
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF-BoxCAM for YOLO Object Detection")
    parser.add_argument("--weights", type=str, required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--target_index", type=int, required=True, help="Raw prediction index to explain")
    parser.add_argument("--output", type=str, default="rf_boxcam_output.jpg", help="Output path for the visualization")
    
    args = parser.parse_args()

    # Initialize and run
    explainer = RFBoxCAM(weights_path=args.weights)
    img, heatmap = explainer.generate(image_path=args.image, target_index=args.target_index)
    explainer.save_visualization(img, heatmap, args.output)
    
    print(f"Saliency map saved successfully to {args.output}")