#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>

void inpaint2(cv::InputArray _src, cv::InputArray _mask, cv::OutputArray _dst, double inpaintRange, int flags)
{
    // Convertir las entradas a matrices de OpenCV
    cv::Mat src = _src.getMat();
    cv::Mat mask = _mask.getMat();

    // Crear la imagen de salida con el mismo tamaño y tipo que la imagen fuente
    _dst.create(src.size(), src.type());
    cv::Mat dst = _dst.getMat();

    // Usar la función de inpainting moderna de OpenCV
    cv::inpaint(src, mask, dst, inpaintRange, flags);
}

