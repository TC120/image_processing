#include <stdio.h>
#include "io_png.h"
#include <string.h>
#include <stdlib.h>
#include <unistd.h> // 在gcc编译器中，使用的头文件因gcc版本的不同而不同

void main()
{
    char *image_path = "/mnt/c/Users/congh2/Documents/vscode_workspaces/image_processing/image_read_save/image/lena.png";
    char *save_path = "/mnt/c/Users/congh2/Documents/vscode_workspaces/image_processing/image_read_save/image/lena_copy.png";
    float *U = NULL;
      /* Size of the input image: N2xN1 matrix */
    size_t N2,N1;/* N2=number of rows (dx2) and N1=number of columns (dx1) */
    size_t Nc;   /* Number of channels */
    size_t NNc;  /* Total size of the image: N2xN1xNc */
    int flag;


    U = io_png_read_f32(image_path, &N1, &N2, &Nc);
    printf("image size is (%d, %d, %d)", (int)N1, (int)N2, (int)Nc);
    flag = io_png_write_f32(save_path, U, (int)N1, (int)N2, (int)Nc);
    printf("flag is %d \n ", flag);

}












