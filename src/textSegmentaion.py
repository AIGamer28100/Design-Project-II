import cv2, numpy as np
import sys
import time


def apply_brightness_contrast(input_img, brightness, contrast):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255 + brightness
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def WordExtract(orig, dilate, count):
    mult = 1.1   # I wanted to show an area slightly larger than my min rectangle set this to one if you don't
    img_box = orig.copy()

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    croppedImages = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > 60000:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            W = rect[1][0]
            H = rect[1][1]

            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]
            x1 = min(Xs)
            x2 = max(Xs)
            y1 = min(Ys)
            y2 = max(Ys)

            rotated = False
            angle = (rect[2])
            if angle > 90:
                angle = angle + (90-angle)
                rotated = True
            else:
                angle = angle + (0-angle)

            if rotated:
                W, H = H, W

            center = (int((x1+x2)/2), int((y1+y2)/2))
            size = (int(mult*(x2-x1)),int(mult*(y2-y1)))

            M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

            cropped = cv2.getRectSubPix(img_box, size, center)
            cropped = cv2.warpAffine(cropped, M, size)

            croppedRotated = cv2.getRectSubPix(cropped, (int(H*mult), int(W*mult)), (size[0]/2, size[1]/2))

            cv2.imwrite(f'./output/{str(count)}.png', croppedRotated)
            print(f"Writing \t :: \t'./output/{str(count)}.png'")
            croppedImages.append(f'./output/{str(count)}.png')

            count = count+1

    return croppedImages, count



def main(path, i):
    print(path)
    # read image
    orig = cv2.imread(path)

    # grey scaling
    grey_img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    # contrasting
    b=50 #brightness_const
    c=120 #contrast_const
    contrast = apply_brightness_contrast(grey_img, b, c)

    # global thresholding
    ret, global_thresh = cv2.threshold(contrast, 50, 255, cv2.THRESH_BINARY)

    # adaptive thresholding
    adapt_thresh = cv2.adaptiveThreshold(global_thresh, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # denoishing
    denoised = cv2.fastNlMeansDenoising(adapt_thresh, 11, 31, 5) #11, 45, 9 #11, 31, 9 #30,7,25

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilate = cv2.dilate(denoised, kernel, iterations=5)

    dilated_blur = cv2.GaussianBlur(dilate, (21, 21), 10)

    ret, dilated_thresh = cv2.threshold(dilated_blur, 50, 255, cv2.THRESH_BINARY)

    dilated_adapted = cv2.adaptiveThreshold(dilated_thresh, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    dilatedfinal = cv2.dilate(dilated_adapted, kernel, iterations=10)
    # cv2.imshow("dilatedfinal", cv2.resize(dilatedfinal, (1000, 1000)))

    output, i = WordExtract(orig, dilatedfinal, i)

    cnts = cv2.findContours(dilatedfinal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]


    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > 50000:
            x, y, w, h = cv2.boundingRect(cnt)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            orig = cv2.drawContours(orig,[box],0,(0,0,255), 10)
            cv2.rectangle(orig, (x, y),(x + w, y + h), (36, 255, 12), 10)

    # cv2.imshow("Testing red", cv2.resize(checking, (1000, 1000)))
    # cv2.imshow("Testing", cv2.resize(orig, (1000, 1000)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if len(output) < 1:
        return None
    else:
        return output, i

if __name__ == '__main__':
    # read image
    main(sys.argv[1])
