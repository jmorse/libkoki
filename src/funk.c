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

static uint32_t koki_funky_integral_image_sum( const uint32_t *ii, int imgwidth,
		const CvRect *region );

static bool
koki_funky_threshold_adaptive_pixel( const IplImage *frame,
				    uint32_t *iimg,
				    const CvRect *roi,
				    uint16_t x, uint16_t y, int16_t c )
{
	uint16_t w, h;
	uint32_t sum;
	uint32_t cmp;

	w = roi->width;
	h = roi->height;

	/* calculate threshold */
	sum = koki_funky_integral_image_sum( iimg, frame->width, roi );

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

#define ii_pix( img, w, x, y ) ( (img)[(((y) % 16) * w) + (x)])

static void
koki_funky_integral_image_advance( uint32_t *ii, int imgwidth,
		                   const IplImage *srcimg,
				   uint32_t *sum, uint16_t width, int y)
{
	uint16_t x;

	for( x=0; x < width; x++ ) {
		uint32_t v = 0;

		sum[x] += KOKI_IPLIMAGE_GS_ELEM( srcimg, x, y );

		v = sum[x];

		if( x > 0 )
			v += ii_pix( ii, imgwidth, x-1, y );

		ii_pix( ii, imgwidth, x, y ) = v;
	}
}

static uint32_t
koki_funky_integral_image_sum( const uint32_t *ii, int imgwidth,
				  const CvRect *region )
{
	uint32_t v;
	/* Coordinates of the south-east pixel of the region */
	const uint32_t se_x = region->x + region->width - 1;
	const uint32_t se_y = region->y + region->height - 1;

	/* SE corner */
	v = ii_pix( ii, imgwidth, se_x, se_y );

	if( region->x > 0 && region->y > 0 )
		/* NW of top left corner */
		v += ii_pix( ii, imgwidth, region->x - 1, region->y - 1 );

	if( region->x > 0 )
		/* E of bottom left corner */
		v -= ii_pix( ii, imgwidth, region->x - 1, se_y );

	if( region->y > 0 )
		/* N of top right corner */
		v -= ii_pix( ii, imgwidth, se_x, region->y - 1 );

	return v;
}

koki_labelled_image_t* koki_funky_label_adaptive( koki_t *koki,
					    const IplImage *frame,
					    uint16_t window_size,
					    int16_t thresh_margin )
{
	uint16_t x, y;
	koki_labelled_image_t *lmg;
	IplImage *thresh_img = NULL;

	/* Instead of an integral image, use a plain array instead. We can
	 * then optimise around this later. Also: the sum array. */
	uint32_t *iimg =
		calloc(1, sizeof(uint32_t) * frame->width * 16);
	uint32_t *sumarr = calloc(1, sizeof(uint32_t) * frame->width);

	assert(frame != NULL && frame->nChannels == 1);

	lmg = koki_labelled_image_new( frame->width, frame->height );

	CvRect win;
	int yadvance = window_size / 2;

	/* Process the first lump of the image */
	for (int i = 0; i < yadvance; i++)
		koki_funky_integral_image_advance(iimg, frame->width,
						   frame, sumarr,
						   frame->width - 1, i);

	int half_win_size = window_size / 2;
	int winx, winy, winwidth, winheight;
	winy = -half_win_size;
	winheight = 1;
	for( y=0; y<frame->height; y++ ) {

		/* Advance the integral image */
		yadvance++;
		yadvance = MIN(yadvance, frame->height - 1);

		/* Skip repeatedly integrating the last line */
		if (yadvance != frame->height || winheight < window_size)
			koki_funky_integral_image_advance( iimg, frame->width,
					frame, sumarr, frame->width - 1,
					yadvance);

		winx = -half_win_size;
		winwidth = 1;

		for( x=0; x<frame->width; x++ ) {
			win.x = MAX(0, winx);
			win.y = MAX(0, winy);
			win.width = winwidth;
			win.height = winheight;

			if( koki_funky_threshold_adaptive_pixel( frame,
							   iimg, &win,
							   x, y, thresh_margin ) ) {
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

			winwidth = MIN(winwidth + 1, window_size);
			winx = MIN(frame->width, winx+1);
		}

		winheight = MIN(winheight + 1, window_size);
		winy = MIN(frame->height, winy+1);
	}

	if( thresh_img != NULL ) {
		koki_log( koki, "thresholded image\n", thresh_img );
		cvReleaseImage( &thresh_img );
	}

	/* Sort out all the remaining labelling related stuff */
	label_image_calc_stats( lmg );

	free(iimg);
	free(sumarr);

	return lmg;
}
