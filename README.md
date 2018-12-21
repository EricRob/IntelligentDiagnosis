## Overall Pipeline

0. Review data for adherence to quality standards, if multi-layered TIFFs run export_top_layer.py
1. Split large images into regular TIFF format (from BigTIFF, run large_image_splitter.py)
2. a. Process image through QuPath to obtain cell locations and clustering information
2. b. Create image binary masks through gimp (gimp_batch_mask)
3. Process images and features into sequences as binary files (preprocess_lstm.py)
4. Run networks (training or testing, recurrence_seq_lstm.py)
5. Summarize training conditions (ID_post_processing.py), or summarize a single testing condition (majority_vote.py)
6. Summarize across multiple tests (multitest_summary.py)