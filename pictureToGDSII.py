#!./venv/bin/python

import cv2
import numpy as np
import gdspy

import os
import sys

try:
    import argcomplete
except:
    argcomplete = None

import argparse


class PictureToGDSII:
    def __init__(self, quiet=False, verbose=False, debug=False):
        self.quiet = quiet
        self.verbose = verbose
        self.debug = debug

    def run_cli(self, prog_name:str, cmd_line_args:list):
        """ Run pictureToGDSII via cmd_line_args """
        # --------------------------------
        # parse arguments
        # --------------------------------
        args = self.parse_args(prog_name, cmd_line_args)
        self._verbose_print(f"Arguments: {args}")

        # --------------------------------
        # run
        # --------------------------------
        self.run(input_file=args.input_file,
                 output_file=args.output_file,
                 layer_num=args.layer_num,
                 pixel_size=args.pixel_size,
                 width_max=args.width_max,
                 height_max=args.height_max,
                 scale=args.scale,
                 dithering=args.dithering,
                 adaptive_threshold=args.adaptive_threshold,
                 masked=args.masked,
                 pixel_cleanup=args.pixel_cleanup,
                 fill_max_distance=args.fill_max_distance)

    def run(self, input_file:str, output_file:str, layer_num:int, pixel_size:float, width_max:float=None, height_max:float=None, scale:float=1.0,
            dithering:str=None, adaptive_threshold:list=None, masked:int=None, pixel_cleanup:str=None, fill_max_distance:float=None):
        """ Run pictureToGDSII """
        # --------------------------------
        # read image
        # --------------------------------
        raw_image = self.read_image(input_file)
        self._print(f"Input file shape: {raw_image.shape[1]} x {raw_image.shape[0]} (w x h [pixels])")

        # --------------------------------
        # scale image
        # --------------------------------
        scaled_image = self.scale_image(raw_image, pixel_size=pixel_size, width_max=width_max, height_max=height_max, scale=scale)
        self._print(f"Scaled file shape: {scaled_image.shape[1]} x {scaled_image.shape[0]} (w x h [pixels])")
        self._debug_save_bmp(f"{output_file}_1_scaled.bmp", scaled_image)

        # --------------------------------
        # convert to gray scale
        # --------------------------------
        gray_scale_image = self.gray_scale_image(scaled_image)
        self._debug_save_bmp(f"{output_file}_2_gray_scale.bmp", gray_scale_image)

        # --------------------------------
        # apply dithering
        # --------------------------------
        dithered_image = gray_scale_image
        if dithering:
            self._print(f"Applying dithering: '{dithering}'")
            dithered_image = self.dither_image(gray_scale_image, dithering)
            self._debug_save_bmp(f"{output_file}_3_dithered.bmp", dithered_image)

        # --------------------------------
        # threshold image
        # --------------------------------
        threshold_image = self.threshold_image(dithered_image, adaptive_threshold)
        self._debug_save_bmp(f"{output_file}_4_threshold.bmp", threshold_image)

        # --------------------------------
        # mask image
        # --------------------------------
        masked_image = threshold_image
        if masked:
            mask = self.mask_image(gray_scale_image, iterations=masked)
            masked_image = np.maximum(mask, threshold_image)
            self._debug_save_bmp(f"{output_file}_5_masked.bmp", masked_image)
            self._debug_save_bmp(f"{output_file}_5_1_mask.bmp", mask)

        # --------------------------------
        # pixel cleanup -> remove diagonal pixels (not manufacturable)
        # --------------------------------
        pixel_clean_image = masked_image
        if pixel_cleanup:
            self._print(f"Removing diagonal pixels")
            pixel_clean_image, debug_pixel_clean_image = self.pixel_clean_image(masked_image, pixel_cleanup)
            stats = {tuple(k):int(v) for k,v in zip(*np.unique(debug_pixel_clean_image.reshape(-1, 3), axis=0, return_counts=True))}
            self._verbose_print(f"Stats: {stats.get((0, 255, 0), 0)} pixels added. {stats.get((0, 0, 255), 0)} pixels removed.")
            self._debug_save_bmp(f"{output_file}_6_pixel_clean.bmp", pixel_clean_image)
            self._debug_save_bmp(f"{output_file}_6_1_pixel_clean_debug.bmp", debug_pixel_clean_image)

        # --------------------------------
        # fill pixels -> add pixels to sparse areas (for manufacturability)
        # --------------------------------
        filled_image = pixel_clean_image
        if fill_max_distance:
            self._print(f"Filling pixels")
            filled_image, debug_filled_image = self.fill_image(pixel_clean_image, pixel_size, fill_max_distance)
            stats = {tuple(k):int(v) for k,v in zip(*np.unique(debug_filled_image.reshape(-1, 3), axis=0, return_counts=True))}
            self._verbose_print(f"Stats: {stats.get((0, 255, 0), 0)} pixels added. {stats.get((0, 0, 255), 0)} pixels removed.")
            self._debug_save_bmp(f"{output_file}_7_filled.bmp", filled_image)
            self._debug_save_bmp(f"{output_file}_7_1_filled_debug.bmp", debug_filled_image)

        # --------------------------------
        # write gds
        # --------------------------------
        self._verbose_print(f"Generating {output_file}.gds")
        created_rects = self.write_gds(filled_image, output_file=output_file, layer_num=layer_num, pixel_size=pixel_size)
        self._print(f"Generated: {output_file}.gds")
        self._verbose_print(f"Stats: {created_rects} rectangles created in {output_file}.gds")

    def parse_args(self, prog_name:str, cmd_line_args:list):
        parser = argparse.ArgumentParser(
            description="Converter for images to GDSII format",
            prog=prog_name,
        )

        # mandatory arguments
        parser.add_argument("input_file", type=str,
                            help="Input image to convert to GDSII.")
        
        # debug arguments
        parser.add_argument("--quiet", "-q", action="store_true",
                            help="Silences all stdout printing.")
        parser.add_argument("--verbose", "-v", action="store_true",
                            help="Give more printed output.")
        parser.add_argument("--debug", action="store_true",
                            help="Output intermediate images.")

        # settings
        parser.add_argument("--output-file", "-o", default="image", type=str,
                            help="Output file name. And the GDSII cells name. Default: 'image'")
        parser.add_argument("--layer-num", "-l", default=1, type=int,
                            help="Layer number of the GDSII file. Default: 1")
        parser.add_argument("--pixel-size", "-p", default=2, type=float,
                            help="Size of each pixel-size rectangle (minium width & space) [um]. Default: 2 [um]")
        parser.add_argument("--width-max", "--width", "-w", type=float,
                            help="Scales the image accordingly to meet (or undershoot) the 'width-max' [um]. Note: If set, 'scale'-argument is ignored. If not set, 'height-max'-argument scales the image proportionally.")
        parser.add_argument("--height-max", "--height", type=float,
                            help="Scales the image accordingly to meet (or undershoot) the 'height-max' [um]. Note: If set, 'scale'-argument is ignored. If not set, 'width-max'-argument scales the image proportionally.")
        parser.add_argument("--scale", "-s", default=1, type=float,
                            help="Scales the input image. Note: Each pixel is the size of 'pixel-size'-argument. Default: 1.0")

        # extra options
        parser.add_argument("--dithering", "-d", choices=["1", "Floyd-Steinberg", "fs", "2", "Jarvis-Judice-Ninke", "jjn", "3", "Stucki", "st", "4", "Atkinson", "a", "5", "Burkes", "b", "6", "Sierra", "s", "7", "Sierra-Filter-Lite", "sfl", "8", "Sierra-Two-Row", "str",
                                                          "9", "Halftone2x2", "h2", "10", "Halftone4x4", "h4", "11", "Bayer2x2", "b2", "12", "Bayer4x4", "b4", "13", "Bayer8x8", "b8"],
                            help="""Applies a specified dithering algorithm.
                                    Possible Algorithms include:
                                    - 1.  Floyd-Steinberg (fs)
                                    - 2.  Jarvis-Judice-Ninke (jjn)
                                    - 3.  Stucki (st)
                                    - 4.  Atkinson (a)
                                    - 5.  Burkes (b)
                                    - 6.  Sierra (s)
                                    - 7.  Sierra-Filter-Lite (sfl)
                                    - 8.  Sierra-Two-Row (str)
                                    - 9.  Halftone2x2 (h2)
                                    - 10. Halftone4x4 (h4)
                                    - 11. Bayer2x2 (b2)
                                    - 12. Bayer4x4 (b4)
                                    - 13. Bayer8x8 (b8)
                            """)
        parser.add_argument("--adaptive-threshold", "--at", nargs=2, type=float, metavar=("KERNEL_SIZE", "OFFSET"),
                            help="Applies an adaptive gaussian threshold when the image is binarized. Default: global threshold")
        parser.add_argument("--masked", "-m", type=int, metavar="EXPANSION_ITERATIONS",
                            help="Applies masking on the image with expansion. Removes spread out pixels.")
        parser.add_argument("--pixel-cleanup", "-c", choices=["remove", "balanced", "random", "add"],
                            help="Determines how to handle diagonal pixels, which are often not manufacturable. 'remove' removes pixels, 'add' adds pixels, 'balanced' balances removal and addition of pixels. 'random' randomly adds or removes critical pixels.")
        parser.add_argument("--fill-max-distance", "-f", default=None, type=float,
                            help="Fill the remaining area with small squares at a given distance in [um].")

        # parse args
        if argcomplete:
            argcomplete.autocomplete(parser, always_complete_options=False)
        args = parser.parse_args(cmd_line_args)
        
        # sanitize arguments
        # output_file
        args.output_file = args.output_file.removesuffix(".gds")
        # adaptive_threshold
        if args.adaptive_threshold:
            if (args.adaptive_threshold[0] - int(args.adaptive_threshold[0])) or \
               (args.adaptive_threshold[0] <= 1) or \
               (args.adaptive_threshold[0] % 2 == 0):
                parser.print_usage()
                print(f"{prog_name}: error: first argument of --adaptive_threshold/-at must be of type 'int', >1 and odd!")
                return
            args.adaptive_threshold[0] = int(args.adaptive_threshold[0])

        # overwrite __init__ flags
        self.quiet = args.quiet
        self.verbose = args.verbose
        self.debug = args.debug

        # return parsed args
        return args

    def read_image(self, file:str) -> np.ndarray:
        return cv2.imread(file)

    def scale_image(self, image:np.ndarray, pixel_size:float, width_max:float=None, height_max:float=None, scale:float=None) -> np.ndarray:
        image_height_pix = image.shape[0]
        image_width_pix  = image.shape[1]
        scale_dsize = None
        if (width_max or height_max):
            # set desired width/height, maintain aspect ratio if only one of width or height is set!
            scale_dsize = (int(width_max/pixel_size)  if width_max  else int(image_width_pix*(int(height_max/pixel_size)/image_height_pix)), 
                           int(height_max/pixel_size) if height_max else int(image_height_pix*(int(width_max/pixel_size)/image_width_pix)))
        return cv2.resize(image,
                          dsize=scale_dsize, # if None -> fx and fy are used
                          fx=scale, # scale is ignored when scale_dsize is set
                          fy=scale  # scale is ignored when scale_dsize is set
                          )

    def gray_scale_image(self, image:np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # jpg & png are BGR ordered

    def dither_image(self, image:np.ndarray, dither_type:str) -> np.ndarray:
        """ Performs error-diffusion or ordered dithering on the input image.
            - 1.  Floyd-Steinberg (fs)
            - 2.  Jarvis-Judice-Ninke (jjn)
            - 3.  Stucki (st)
            - 4.  Atkinson (a)
            - 5.  Burkes (b)
            - 6.  Sierra (s)
            - 7.  Sierra-Filter-Lite (sfl)
            - 8.  Sierra-Two-Row (str)
            - 9.  Halftone2x2 (h2)
            - 10. Halftone4x4 (h4)
            - 11. Bayer2x2 (b2)
            - 12. Bayer4x4 (b4)
            - 13. Bayer8x8 (b8)
        """
        # setting the dither_matrix -> "-0" represents the pixel being operated on
        if (dither_type == "Floyd-Steinberg") or (dither_type == "fs") or (dither_type == "1"):
            dither_matrix = np.array([
                [   0,   -0, 7/16],
                [3/16, 5/16, 1/16]])
            return self.dither_error_diffusion(image=image, dither_matrix=dither_matrix)
        elif (dither_type == "Jarvis-Judice-Ninke") or (dither_type == "jjn") or (dither_type == "2"):
            dither_matrix = np.array([
                [   0,    0,   -0, 7/48, 5/48],
                [3/48, 5/48, 7/48, 5/48, 3/48],
                [1/48, 3/48, 5/48, 3/48, 1/48]])
            return self.dither_error_diffusion(image=image, dither_matrix=dither_matrix)
        elif (dither_type == "Stucki") or (dither_type == "st") or (dither_type == "3"):
            dither_matrix = np.array([
                [   0,    0,   -0, 8/42, 4/42],
                [2/42, 4/42, 8/42, 4/42, 2/42],
                [1/42, 2/42, 4/42, 2/42, 1/42]])
            return self.dither_error_diffusion(image=image, dither_matrix=dither_matrix)
        elif (dither_type == "Atkinson") or (dither_type == "a") or (dither_type == "4"):
            dither_matrix = np.array([
                [  0,   0,  -0, 1/8, 1/8],
                [  0, 1/8, 1/8, 1/8,   0],
                [  0,   0, 1/8,   0,   0]]) # only support 2x3 or 3x5 matrix
            return self.dither_error_diffusion(image=image, dither_matrix=dither_matrix)
        elif (dither_type == "Burkes") or (dither_type == "b") or (dither_type == "5"):
            dither_matrix = np.array([
                [   0,    0,   -0, 8/32, 4/32],
                [2/32, 4/32, 8/32, 4/32, 2/32],
                [   0,    0,    0,    0,    0]]) # only support 2x3 or 3x5 matrix
            return self.dither_error_diffusion(image=image, dither_matrix=dither_matrix)
        elif (dither_type == "Sierra") or (dither_type == "s") or (dither_type == "6"):
            dither_matrix = np.array([
                [   0,    0,   -0, 4/16, 3/16],
                [1/16, 2/16, 3/16, 2/16, 1/16],
                [   0,    0,    0,    0,    0]]) # only support 2x3 or 3x5 matrix
            return self.dither_error_diffusion(image=image, dither_matrix=dither_matrix)
        elif (dither_type == "Sierra-Filter-Lite") or (dither_type == "sfl") or (dither_type == "7"):
            dither_matrix = np.array([
                    [  0,  -0, 2/4],
                    [1/4, 1/4,   0]])
            return self.dither_error_diffusion(image=image, dither_matrix=dither_matrix)
        elif (dither_type == "Sierra-Two-Row") or (dither_type == "str") or (dither_type == "8"):
            dither_matrix = np.array([
                [   0,    0,   -0, 5/32, 3/32],
                [2/32, 4/32, 5/32, 4/32, 2/32],
                [   0, 2/32, 3/32, 2/32,    0]])
            return self.dither_error_diffusion(image=image, dither_matrix=dither_matrix)
        elif (dither_type == "Halftone2x2") or (dither_type == "h2") or (dither_type == "9"):
            dither_matrix_array = np.array([
                [[0,0],[0,0]],
                [[255,0],[0,0]],
                [[255,255],[0,0]],
                [[255,255],[255,0]],
                [[255,255],[255,255]]
            ])
            return self.dither_halftone(image=image, dither_matrix_array=dither_matrix_array)
        elif (dither_type == "Halftone4x4") or (dither_type == "h4") or (dither_type == "10"):
            dither_matrix_array = np.array([
                [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                [[0,0,0,0],[0,255,0,0],[0,0,0,0],[0,0,0,0]],
                [[0,0,0,0],[0,255,255,0],[0,0,0,0],[0,0,0,0]],
                [[0,0,0,0],[0,255,255,0],[0,255,0,0],[0,0,0,0]],
                [[0,0,0,0],[0,255,255,0],[0,255,255,0],[0,0,0,0]],
                [[0,255,0,0],[0,255,255,0],[0,255,255,0],[0,0,0,0]],
                [[0,255,0,0],[0,255,255,255],[0,255,255,0],[0,0,0,0]],
                [[0,255,0,0],[0,255,255,0],[0,255,255,0],[0,0,255,0]],
                [[0,255,0,0],[0,255,255,255],[255,255,255,0],[0,0,255,0]],
                [[0,255,255,0],[0,255,255,255],[255,255,255,0],[0,0,255,0]],
                [[0,255,255,0],[0,255,255,255],[255,255,255,255],[0,0,255,0]],
                [[0,255,255,0],[0,255,255,255],[255,255,255,255],[0,255,255,0]],
                [[0,255,255,0],[255,255,255,255],[255,255,255,255],[0,255,255,0]],
                [[0,255,255,255],[255,255,255,255],[255,255,255,255],[0,255,255,0]],
                [[0,255,255,255],[255,255,255,255],[255,255,255,255],[0,255,255,255]],
                [[0,255,255,255],[255,255,255,255],[255,255,255,255],[255,255,255,255]],
                [[255,255,255,255],[255,255,255,255],[255,255,255,255],[255,255,255,255]]
            ])
            return self.dither_halftone(image=image, dither_matrix_array=dither_matrix_array)
        elif (dither_type == "Bayer2x2") or (dither_type == "b2") or (dither_type == "11"):
            dither_matrix = np.array([
                [0, 2],
                [3, 1]
            ]) / 4.0
            return self.dither_bayer(image=image, dither_matrix=dither_matrix)
        elif (dither_type == "Bayer4x4") or (dither_type == "b4") or (dither_type == "12"):
            dither_matrix = np.array([
                [ 0,  8,  2, 10],
                [12,  4, 14,  6],
                [ 3, 11,  1,  9],
                [15,  7, 13,  5]
            ]) / 16.0
            return self.dither_bayer(image=image, dither_matrix=dither_matrix)
        elif (dither_type == "Bayer8x8") or (dither_type == "b8") or (dither_type == "13"):
            dither_matrix = np.array([
                [ 0, 32,  8, 40,  2, 34, 10, 42],
                [48, 16, 56, 24, 50, 18, 58, 26],
                [12, 44,  4, 36, 14, 46,  6, 38],
                [60, 28, 52, 20, 62, 30, 54, 22],
                [ 3, 35, 11, 43,  1, 33,  9, 41],
                [51, 19, 59, 27, 49, 17, 57, 25],
                [15, 47,  7, 39, 13, 45,  5, 37],
                [63, 31, 55, 23, 61, 29, 53, 21]
            ]) / 64.0
            return self.dither_bayer(image=image, dither_matrix=dither_matrix)
        else:
            raise ValueError(f"dither_image: Filter '{dither_type}' not supported!")

    def dither_error_diffusion(self, image:np.ndarray, dither_matrix:np.ndarray) -> np.ndarray:
        # allow for negative values and higher than 255 pixel values
        image_out = image.astype(np.int16)
        width  = image_out.shape[1]
        height = image_out.shape[0]
        # supported dither_matrix shapes
        is2x3Matrix = dither_matrix.shape == (2, 3)
        is3x5Matrix = dither_matrix.shape == (3, 5)
        # loop through image
        for (y,x),pixel in np.ndenumerate(image_out):
            # quantize pixel
            quantized_pixel = 0 if (pixel < 127) else 255
            image_out[y][x] = quantized_pixel
            # forwarding the quantization error
            quantization_error = pixel - quantized_pixel
            # dont propagate error 'outside' the image
            if is2x3Matrix:
                if (x != width-1):                       image_out[y][x+1]   += quantization_error * dither_matrix[0][2]
                if ((x != 0) and (y != height-1)):       image_out[y+1][x-1] += quantization_error * dither_matrix[1][0]
                if (y != height-1):                      image_out[y+1][x]   += quantization_error * dither_matrix[1][1]
                if ((x != width-1) and (y != height-1)): image_out[y+1][x+1] += quantization_error * dither_matrix[1][2]
            elif is3x5Matrix:
                if (x < width-1):                      image_out[y][x+1]   += quantization_error * dither_matrix[0][3]
                if (x < width-2):                      image_out[y][x+2]   += quantization_error * dither_matrix[0][4]
                if ((x > 1) and (y < height-1)):       image_out[y+1][x-2] += quantization_error * dither_matrix[1][0]
                if ((x > 0) and (y < height-1)):       image_out[y+1][x-1] += quantization_error * dither_matrix[1][1]
                if (y < height-1):                     image_out[y+1][x]   += quantization_error * dither_matrix[1][2]
                if ((x < width-1) and (y < height-1)): image_out[y+1][x+1] += quantization_error * dither_matrix[1][3]
                if ((x < width-2) and (y < height-1)): image_out[y+1][x+2] += quantization_error * dither_matrix[1][4]
                if ((x > 1) and (y < height-2)):       image_out[y+2][x-2] += quantization_error * dither_matrix[2][0]
                if ((x > 0) and (y < height-2)):       image_out[y+2][x-1] += quantization_error * dither_matrix[2][1]
                if (y < height-2):                     image_out[y+2][x]   += quantization_error * dither_matrix[2][2]
                if ((x < width-1) and (y < height-2)): image_out[y+2][x+1] += quantization_error * dither_matrix[2][3]
                if ((x < width-2) and (y < height-2)): image_out[y+2][x+2] += quantization_error * dither_matrix[2][4]
            else:
                raise ValueError(f"dither_error_diffusion: dither_matrix shape ({dither_matrix.shape}) not supported!")
        # return uint8 image
        return image_out.astype(np.uint8)

    def dither_halftone(self, image:np.ndarray, dither_matrix_array:np.ndarray) -> np.ndarray:
        # does a copy of image
        image_out = image.astype(np.uint8)
        width  = image_out.shape[1]
        height = image_out.shape[0]
        # prepare values
        dither_matrix_width  = dither_matrix_array.shape[2]
        dither_matrix_height = dither_matrix_array.shape[1]
        values               = dither_matrix_array.shape[0]
        # loop through image
        for y in range(0, height, dither_matrix_height):
            for x in range(0, width, dither_matrix_width):
                # get affected pixels
                img_slice = image_out[y:y+dither_matrix_height, x:x+dither_matrix_width]
                # calculate intensity
                average_intensity = img_slice.mean()/255
                # get closest color
                color_idx = int(average_intensity * (values-1))
                closest_color = dither_matrix_array[color_idx]
                # make sure closest_color is of the same shape (especially for edges!!)
                closest_color = closest_color[0:img_slice.shape[0], 0:img_slice.shape[1]]
                # write image
                image_out[y:y+dither_matrix_height, x:x+dither_matrix_width] = closest_color
        # return uint8 image
        return image_out

    def dither_bayer(self, image:np.ndarray, dither_matrix:np.ndarray) -> np.ndarray:
        # does a copy of image
        image_out = image.astype(np.uint8)
        # loop through image
        for (y,x),pixel in np.ndenumerate(image_out):
            # compute threshold
            threshold = dither_matrix[y % dither_matrix.shape[1], x % dither_matrix.shape[0]]*128
            # quantize pixel
            image_out[y][x] = 0 if (pixel < threshold) else 255
        # return uint8 image
        return image_out

    def threshold_image(self, image:np.ndarray, adaptive_threshold:list=None) -> np.ndarray:
        if adaptive_threshold:
            return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptive_threshold[0], adaptive_threshold[1])
        else:
            return cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)[1]

    def mask_image(self, image:np.ndarray, iterations:int=0) -> np.ndarray:
        threshold_image = self.threshold_image(image)
        kernel = np.ones((3,3), np.uint8)
        return cv2.erode(threshold_image, kernel, iterations=iterations)

    def pixel_clean_image(self, image:np.ndarray, method:str=None) -> np.ndarray:
        """ Removes diagonal pixels in the image by adding or removing pixels.
            - in debug_out: added   pixels are green (0, 255, 0)
            - in debug_out: removed pixels are red   (0, 0, 255)

            The "method" describes how to overcome diagonal pixels:
                - "remove": only removes pixels -> tends towards "whiter" images
                - "balanced": removes and adds pixels in alternating fashion
                - "random": randomly decides to add or remove pixels
                - "add": only adds pixels -> tends towards "blacker" images
                - otherwise -> fails silently
        """
        image_out = np.copy(image)
        debug_out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # prepare balanced pixel cleaning
        balanced_add = True
        # prepare random pixel cleaning
        np.random.seed(0)
        # loop through image
        for (y,x),value in np.ndenumerate(image_out):
            # skip last row and column
            if (x != image_out.shape[1]-1) and (y != image_out.shape[0]-1):
                if value == 0 and image_out[y][x+1] and image_out[y+1][x] and image_out[y+1][x+1] == 0: # backward \ - pixels
                    # prepare random number
                    random_add = bool(np.random.choice(a=[True, False]))
                    if (method == "add") or ((method == "balanced") and (balanced_add)) or ((method == "random") and (random_add)):
                        image_out[y+1][x] = 0
                        debug_out[y+1][x] = [0, 255, 0] # add green pixel when adding pixels to output
                    else: #elif (method == "remove") or ((method == "balanced") and (not balanced_add)) or ((method == "random") and (not random_add)):
                        image_out[y+1][x+1] = 255
                        debug_out[y+1][x+1] = [0, 0, 255] # add red pixel when removing pixels from output
                elif value and image_out[y][x+1] == 0 and image_out[y+1][x] == 0 and image_out[y+1][x+1]: # forward / - pixels
                    # prepare random number
                    random_add = bool(np.random.choice(a=[True, False]))
                    if (method == "add") or ((method == "balanced") and (balanced_add)) or ((method == "random") and (random_add)):
                        image_out[y+1][x+1] = 0 # kann dadurch fehler erstellen!!! fix it
                        debug_out[y+1][x+1] = [0, 255, 0] # place green in debug when adding pixels
                    else: #elif (method == "remove") or ((method == "balanced") and (not balanced_add)) or ((method == "random") and (not random_add)):
                        image_out[y+1][x] = 255
                        debug_out[y+1][x] = [0, 0, 255] # add red pixel when removing pixels from output
                # flip balanced_add_remove
                balanced_add = not balanced_add

        return image_out, debug_out

    def fill_image(self, image:np.ndarray, pixel_size:float, max_distance:float) -> np.ndarray:
        """ Fills the image with pixels for manufacturability.
            - in debug_out: added   pixels are green (0, 255, 0)

            pixel_size is given in um.
            max_distance is given in um.
        """
        image_out = np.copy(image)
        debug_out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # calculations
        height, width = image_out.shape
        max_pixel_distance = int(max_distance/pixel_size)
        # loop through image
        for y in range(height-max_pixel_distance):
            for x in range(width-max_pixel_distance):
                # check square for pixels
                no_pixel = True
                for local_y in range(max_pixel_distance):
                    for local_x in range(max_pixel_distance):
                        if (image_out[y+local_y][x+local_x] == 0): # found a colored pixel
                            no_pixel = False
                            break
                    if not no_pixel:
                        break
                if no_pixel:
                    image_out[int(y+(max_pixel_distance/2))][int(x+(max_pixel_distance/2))] = 0
                    debug_out[int(y+(max_pixel_distance/2))][int(x+(max_pixel_distance/2))] = [0, 255, 0] # place green in debug when adding pixels

        return image_out, debug_out

    def write_gds(self, image:np.ndarray, output_file:str, layer_num:int, pixel_size:float):
        # create cell
        lib = gdspy.GdsLibrary()
        cell = lib.new_cell(os.path.basename(output_file))

        # get hight of image for coordinate conversion
        height = image.shape[0]

        # iterate image
        count_rects= 0
        for (y,x),value in np.ndenumerate(image):
            # draw only black pixels
            if value == 0:
                rect = gdspy.Rectangle([(x)  *pixel_size,(height-y-1)*pixel_size],
                                       [(x+1)*pixel_size,(height-y)  *pixel_size],
                                       layer=layer_num)
                cell.add(rect)
                count_rects += 1

        # write gds
        lib.write_gds(f"{output_file}.gds")

        # return created rects
        return count_rects

    def _print(self, s:str):
        if not self.quiet:
            print(s)

    def _verbose_print(self, s:str):
        if self.verbose and not self.quiet:
            print(f"VERBOSE: {s}")

    def _debug_save_bmp(self, file:str, image:np.ndarray):
        if self.debug:
            file = file.removesuffix(".bmp")
            cv2.imwrite(f"{file}.bmp", image)
            if not self.quiet:
                print(f"DEBUG: Saved {file}.bmp")


# ----
# main
# ----
if __name__ == "__main__":
    prog_name = os.path.basename(sys.argv[0])
    args      = sys.argv[1:]
    PictureToGDSII().run_cli(prog_name, args)
