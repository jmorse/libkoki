#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#include <cv.h>
#include "labelling.h"
#include "threshold.h"
#include "integral-image.h"

#define label_aliases_index( arr, index ) g_array_index( arr, label_t, index )
#define label_clips_index( arr, index ) g_array_index( arr, koki_clip_region_t, index )

static label_t label_find_canonical( koki_labelled_image_t *lmg,
				     label_t l )
{
	while(1) {
		label_t a = label_aliases_index( lmg->aliases, l-1 );

		if( a == l )
			/* Found the lowest alias */
			return a;

		l = a;
	}
}

static void label_alias( koki_labelled_image_t *lmg, label_t l_canon, label_t l_alias )
{
	label_t *l;

	/* Resolve to the minimum alias of l_alias */
	l_alias = label_find_canonical( lmg, l_alias );
	l_canon = label_find_canonical( lmg, l_canon );

	/* Alias l_alias to l_canon */
	l = &label_aliases_index( lmg->aliases, l_alias-1 );
	*l = l_canon;
}

static void set_label(koki_labelled_image_t *labelled_image,
		      uint16_t x, uint16_t y, label_t label)
{

	if (label == 0){

		KOKI_LABELLED_IMAGE_LABEL(labelled_image, x, y) = 0;

	} else {

		assert(labelled_image->aliases != NULL
		       && labelled_image->aliases->len > label-1);

		KOKI_LABELLED_IMAGE_LABEL(labelled_image, x, y)
			= label_aliases_index( labelled_image->aliases,
					       label-1 );
	}

}

static void label_dark_pixel( koki_labelled_image_t *lmg,
			      uint16_t x, uint16_t y )
{
	label_t label_tmp;

	/* if pixel above is labelled, join that label */
	label_tmp = get_connected_label(lmg, x, y, N);
	if (label_tmp > 0){
		set_label(lmg, x, y, label_tmp);
		return;
	}

	/* if NE pixel is labelled, some merging may need to occur */
	label_tmp = get_connected_label(lmg, x, y, NE);
	if (label_tmp > 0){

		label_t label_w, label_nw, l1, l2, label_min, label_max;
		label_w  = get_connected_label(lmg, x, y, W);
		label_nw = get_connected_label(lmg, x, y, NW);

		/* if one of the pixels W or NW are labelled, they should
		   be merged together */
		if (label_w > 0 || label_nw > 0){

			l1 = label_aliases_index( lmg->aliases, label_tmp-1 );

			l2 = label_nw > 0
				? label_aliases_index( lmg->aliases, label_nw-1 )
				: label_aliases_index( lmg->aliases, label_w-1 );

			/* identify lowest label */
			label_max = l2;
			label_min = l1;
			if (label_max < label_min){
				label_min = l2;
				label_max = l1;
			}

			set_label(lmg, x, y, label_min);
			label_alias( lmg, label_min, label_max );
		} else {

			set_label(lmg, x, y, label_tmp);

		}

		return;
	}

	/* Otherwise, take the NW label, if present */
	label_tmp = get_connected_label(lmg, x, y, NW);
	if (label_tmp > 0){
		set_label(lmg, x, y, label_tmp);
		return;
	}

	/* Otherwise, take the W label, if present */
	label_tmp = get_connected_label(lmg, x, y, W);
	if (label_tmp > 0){
		set_label(lmg, x, y, label_tmp);
		return;
	}

	/* If we get this far, a new region has been found */

	/* Check we do not exceed the maximum number of labels */
	assert( lmg->aliases->len != KOKI_LABEL_MAX );

	label_tmp = lmg->aliases->len + 1;
	g_array_append_val(lmg->aliases, label_tmp);
	set_label(lmg, x, y, label_tmp);
}

static void label_image_calc_stats( koki_labelled_image_t *labelled_image )
{
	/* Now renumber all labels to ensure they're all canonical */
	for( label_t i=1; i<labelled_image->aliases->len; i++ ) {
		label_t *a = &label_aliases_index( labelled_image->aliases, i-1 );

		*a = label_find_canonical( labelled_image, i );
	}

	/* collect label statistics (mass, bounding box) */

	label_t max_alias = 0;
	GArray *aliases, *clips;

	aliases = labelled_image->aliases;
	clips = labelled_image->clips;

	/* find largest alias */
	for (label_t i=0; i<aliases->len; i++){
		label_t alias = label_aliases_index( aliases, i );
		if (alias > max_alias)
			max_alias = alias;
	}

	/* init clips */
	for (label_t i=0; i<max_alias; i++){
		koki_clip_region_t clip;
		clip.mass = 0;
		clip.max.x = 0;
		clip.max.y = 0;
		clip.min.x = 0xFFFF; /* max out so below works */
		clip.min.y = 0xFFFF;
		g_array_append_val(clips, clip);
	}

	/* gather stats */
	for (uint16_t y=0; y<labelled_image->h; y++){
		for (uint16_t x=0; x<labelled_image->w; x++){

			label_t label, alias;
			koki_clip_region_t *clip;

			label = KOKI_LABELLED_IMAGE_LABEL(labelled_image, x, y);

			/* a threshold white pixel, ignore */
			if (label == 0)
				continue;

			alias = label_aliases_index( aliases, label-1 );
			clip = &label_clips_index( clips, alias-1 );

			clip->mass++;
			if (x > clip->max.x)
				clip->max.x = x;
			if (y > clip->max.y)
				clip->max.y = y;
			if (x < clip->min.x)
				clip->min.x = x;
			if (y < clip->min.y)
				clip->min.y = y;

		}//for col
	}//for row
}


static bool
koki_funky_threshold_adaptive_pixel( const IplImage *frame,
				    const koki_integral_image_t *iimg,
				    const CvRect *roi,
				    uint16_t x, uint16_t y, int16_t c )
{
	uint16_t w, h;
	uint32_t sum;
	uint32_t cmp;

	w = roi->width;
	h = roi->height;

	/* calculate threshold */
	sum = koki_integral_image_sum( iimg, roi );

	/* The following is a rearranged version of
	      threshold = sum / (w*h);
	      if( KOKI_IPLIMAGE_GS_ELEM(frame, x, y) > (threshold-c) ) ...
	   This is re-arranged to avoid division. */

	cmp = KOKI_IPLIMAGE_GS_ELEM(frame, x, y) + c;
	cmp *= w * h;

	/* apply threshold */
	if( cmp > sum )
		return true;

	return false;
}

static void update_pixel( koki_integral_image_t *ii,
			  uint16_t x, uint16_t y )
{
	uint32_t v = 0;

	/* Note that we expect the source image to be greyscale */
	ii->sum[x] += KOKI_IPLIMAGE_GS_ELEM( ii->src, x, y );

	v = ii->sum[x];

	if( x > 0 )
		v += koki_integral_image_pixel( ii, x-1, y );

	koki_integral_image_pixel( ii, x, y ) = v;
}

static void
koki_funky_integral_image_advance( koki_integral_image_t *ii,
				  uint16_t target_x, uint16_t target_y )
{
	uint16_t x, y;
	assert( target_x < ii->w );
	assert( target_y < ii->h );

	/* Advance in the x-direction, but not y first */
	for( x = ii->complete_x; x <= target_x; x++ )
		for( y=0; y < ii->complete_y; y++ )
			update_pixel( ii, x, y );
	ii->complete_x = target_x + 1;

	/* Now advance in the y-direction */
	for( x=0; x < ii->complete_x; x++ )
		for( y = ii->complete_y; y <= target_y; y++ )
			update_pixel( ii, x, y );
	ii->complete_y = target_y + 1;
}

#define ii_pix( img, x, y ) koki_integral_image_pixel( img, x, y )

static uint32_t
koki_funky_integral_image_sum( const koki_integral_image_t *ii,
				  const CvRect *region )
{
	uint32_t v;
	/* Coordinates of the south-east pixel of the region */
	const uint32_t se_x = region->x + region->width - 1;
	const uint32_t se_y = region->y + region->height - 1;

	assert( region->x < ii->complete_x );
	assert( region->y < ii->complete_y );

	/* SE corner */
	v = ii_pix( ii, se_x, se_y );

	if( region->x > 0 && region->y > 0 )
		/* NW of top left corner */
		v += ii_pix( ii, region->x - 1, region->y - 1 );

	if( region->x > 0 )
		/* E of bottom left corner */
		v -= ii_pix( ii, region->x - 1, se_y );

	if( region->y > 0 )
		/* N of top right corner */
		v -= ii_pix( ii, se_x, region->y - 1 );

	return v;
}

static void
koki_funky_threshold_adaptive_calc_window( const IplImage *frame,
					  CvRect *win,
					  uint16_t window_size,
					  uint16_t x, uint16_t y )
{
	uint16_t width, height;
	assert(window_size % 2 == 1);

	width = frame->width;
	height = frame->height;
	assert( x < width);
	assert( y < height);

	/* identify the window - x */
	if (x >= window_size / 2 &&
	    x < (width-1) - window_size / 2){

		/* normal case, away from frame edges */
		win->x      = x - window_size/2;
		win->width  = window_size;

	} else {

		/* we're at the edge, limit roi accordingly */
		win->width = window_size / 2 + 1;
		win->x = x < window_size / 2
			? 0
			: (width-1) - window_size / 2;

	}

	/* identify the window - y */
	if (y >= window_size / 2 &&
	    y < (height-1) - window_size / 2){

		/* normal case, away from frame edges */
		win->y      = y - window_size/2;
		win->height = window_size;

	} else {

		/* we're at the edge, limit roi accordingly */
		win->height = window_size / 2 + 1;
		win->y = y < window_size / 2
			? 0
			: (height-1) - window_size / 2;

	}
}

koki_labelled_image_t* koki_funky_label_adaptive( koki_t *koki,
					    const IplImage *frame,
					    uint16_t window_size,
					    int16_t thresh_margin )
{
	uint16_t x, y;
	koki_integral_image_t *iimg;
	koki_labelled_image_t *lmg;
	IplImage *thresh_img = NULL;

	assert(frame != NULL && frame->nChannels == 1);

	iimg = koki_integral_image_new( frame, false );
	lmg = koki_labelled_image_new( frame->width, frame->height );

	for( y=0; y<frame->height; y++ )
		for( x=0; x<frame->width; x++ ) {
			CvRect win;

			/* Get the ROI from the thresholder */
			koki_funky_threshold_adaptive_calc_window( frame, &win,
							     window_size, x, y );

			/* Advance the integral image */
			if( x == 0 )
				koki_funky_integral_image_advance( iimg,
							     frame->width - 1,
							     win.y + win.height - 1 );

			if( koki_funky_threshold_adaptive_pixel( frame,
							   iimg,
							   &win, x, y, thresh_margin ) ) {
				/* Nothing exciting */
				set_label( lmg, x, y, 0);

				if( thresh_img != NULL )
					KOKI_IPLIMAGE_GS_ELEM( thresh_img, x, y ) = 0xff;
			} else {
				/* Label the thing */
				label_dark_pixel( lmg, x, y );

				if( thresh_img != NULL )
					KOKI_IPLIMAGE_GS_ELEM( thresh_img, x, y ) = 0;
			}
		}

	if( thresh_img != NULL ) {
		koki_log( koki, "thresholded image\n", thresh_img );
		cvReleaseImage( &thresh_img );
	}

	/* Sort out all the remaining labelling related stuff */
	label_image_calc_stats( lmg );

	koki_integral_image_free( iimg );

	return lmg;
}
