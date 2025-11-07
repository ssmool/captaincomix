from captain_cominx.captomix import *

# Optional: Provide a convenience CLI entry

if __name__ == '__main__':
    import argparse
    import json

    p = argparse.ArgumentParser(description='captomix: merge images with mp4/gif layers')
    p.add_argument('input_image')
    p.add_argument('--out', '-o', default='captomix_output')
    p.add_argument('--max_videos', type=int, default=2)
    p.add_argument('--filters', nargs='*', default=['BRIGHTNESS:1.05', 'CONTRAST:1.1'])
    args = p.parse_args()

    res = captomix(args.input_image, args.out, max_videos=args.max_videos, filters=args.filters)
    print(json.dumps(res, indent=2))

#or usage
#res = captomix(image_gile,save_out_mp4,max_search,filters_fx)
