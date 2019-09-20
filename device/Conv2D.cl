/**********************************************
function: 2D Convolution of CNN
**********************************************/

__kernel void Conv2D( __global float * image_in,  //image input
                      __global float * filter_in, //filter input
                               int filter_kernel_width,		    //filter kernel width
                               int filter_kernel_height,		    //filter kernel height
                               int filter_kernel_depth,		    //filter kernel depth
                               int filter_kernel_num,		    //filter kernel num
                               int image_width,
                               int image_height,
                               int image_depth,
                               int image_num,
                               int conv_width_step,		    //conv width step
                               int conv_height_step,		    //conv height step
                      __global float * image_out) //feature map output
{
	int image_padded_width;      //padded image width
	int image_padded_height;      //padded image height

        int image_new_height;
        int image_new_width;
        int image_pad_width;
        int image_pad_height;

	int x;		 //global id x 
	int y;		 //global id y
        int z;       //global id z
	int ki, kj, kn, kz;	 //filter coordinate,(kj, ki, kn)

	x = get_global_id(0);
	y = get_global_id(1);
	z = get_global_id(2);

        image_new_width = (image_width + conv_width_step -1)/conv_width_step;   //image new width after conv and ceil 
        image_new_height = (image_height + conv_height_step -1)/conv_height_step;    //image new height after conv and ceil
	image_pad_width = (image_new_width - 1)*conv_width_step + filter_kernel_width - image_width; // should add image_pad_width pixel at width
	image_pad_height = (image_new_height - 1)*conv_height_step + filter_kernel_height - image_height; //should add image_pad_height oixal at height

	image_padded_width = image_width + image_pad_width;
	image_padded_height = image_width + image_pad_height;
	
        for(kz = 0; kz < image_num; kz ++)
        {
	    float sum = 0.0;
            for(kn = 0; kn < filter_kernel_depth; kn++)
            {
       	            for(kj = 0; kj < filter_kernel_height; kj++)
                    {
                        for(ki = 0; ki < filter_kernel_width; ki++)
                	    {
                             sum = sum + image_in[kz*image_padded_width*image_padded_height*image_depth + kn*image_padded_width*image_padded_height + y*conv_height_step*image_padded_width + x*conv_width_step + kj*image_padded_width + ki]*filter_in[z*filter_kernel_width*filter_kernel_height*filter_kernel_depth + kn*filter_kernel_width*filter_kernel_height + kj*filter_kernel_width + ki];
                        }
                    }
                }
            image_out[kz*image_new_height*image_new_width*filter_kernel_num + y*image_new_width + z*image_new_width*image_new_height + x] = sum;
        }
}
