
;gimp -i -b '(batch-mask "*.tif" 217)' -b '(gimp-quit 0)'
; Name this script 'batch-mask.scm' and save into your GIMP scripting path
(define (batch-mask pattern lower)
  (let* ((filelist (cadr (file-glob pattern 1))))
    (while (not (null? filelist))
           (let* ((filename (car filelist))
                  (image (car (gimp-file-load RUN-NONINTERACTIVE
                                              filename filename)))
             (drawable (car (gimp-image-get-active-layer image))))
             (gimp-threshold (car  (gimp-image-get-active-drawable image)) lower 255)
             (file-tiff-save RUN-NONINTERACTIVE image drawable (string-append "mask_" filename) (string-append "mask_"filename ) 0)
             (gimp-image-delete image))
           (set! filelist (cdr filelist)))))
