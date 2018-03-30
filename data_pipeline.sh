gimp -i -b '(batch-mask "*.tif" 217)' -b '(gimp-quit 0)'
mv mask* /home/wanglab/Desktop/recurrence_seq_lstm/image_data/masks
mv * /home/wanglab/Desktop/recurrence_seq_lstm/image_data/original_images
python3 /home/wanglab/Desktop/recurrence_seq_lstm/IntelligentDiagnosis/image_processor.py