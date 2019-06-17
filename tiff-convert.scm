;gimp -i -b '(tiff-convert "*.tif")' -b '(gimp-quit 0)'
(define (tiff-convert pattern)
  (let* ((filelist (cadr (file-glob pattern 1))))
    (while (not (null? filelist))
           (let* ((filename (car filelist))
                  (image (car (gimp-file-load RUN-NONINTERACTIVE
                                              filename filename)))
             (drawable (car (gimp-image-get-active-layer image))))
             (file-tiff-save RUN-NONINTERACTIVE image drawable (string-append "gimp_" filename) (string-append "gimp_"filename ) 0)
             (gimp-image-delete image))
           (set! filelist (cdr filelist)))))
