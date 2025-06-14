SCANNET200_CLASS_NAMES = ['wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair',
              'bed', 'pillow', 'sink', 'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book',
              'armchair', 'coffee table', 'box', 'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand',
              'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard',
              'bag', 'backpack', 'toilet paper', 'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet',
              'stairs', 'microwave', 'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board',
              'washing machine', 'mirror', 'copier', 'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper',
              'person', 'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail',
              'radiator', 'recycling bin', 'container', 'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light',
              'laundry basket', 'pipe', 'clothes dryer', 'guitar', 'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder',
              'bathroom stall', 'shower wall', 'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat',
              'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', 'fireplace', 'soap dish', 'kitchen counter', 'doorframe', 'toilet paper dispenser',
              'mini fridge', 'fire extinguisher', 'ball', 'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar',
              'ledge', 'toaster oven', 'mouse', 'toilet seat cover dispenser', 'furniture', 'cart', 'storage container', 'scale', 'tissue box', 'light switch',
              'crate', 'power outlet', 'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner', 'candle', 'plunger', 'stuffed animal',
              'headphones', 'dish rack', 'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar', 'purse',
              'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand', 'projector screen', 'divider',
              'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity', 'closet wall', 'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell',
              'stair rail', 'tube', 'bathroom cabinet', 'cd case', 'closet rod', 'coffee kettle', 'structure', 'shower head', 'keyboard piano', 'case of water bottles',
              'coat rack', 'storage organizer', 'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant', 'luggage', 'mattress']

SCANNET200_VALID_CATEGORY_NAME = ['pillow', 'monitor', 'door', 'lamp', 'printer', 'towel',
                                  'tv', 'nightstand', 'plant', 'keyboard', 'backpack', 'printer',
                                  'trash bin', 'trash can', 'recycling bin', 'closet', 'microwave', 'fan',
                                  'washing machine', 'mirror', 'piano', 'radiator', 'telephone', 'clock',
                                  'guitar', 'bicycle', 'oven', 'mouse']

SCANNET200_CLASS_REMAPPER = {
    'trash bin': ['trash bin', 'trash can', 'recycling bin'],
    'computer mouse': ['mouse'],
}

SCANNET200_CLASS_REMAPPER_LIST = ['trash bin', 'trash can', 'recycling bin', 'mouse']

SCANNET200_VALID_CATEGORY_IDX = [SCANNET200_CLASS_NAMES.index(name) for name in SCANNET200_VALID_CATEGORY_NAME]


def remap_categories(obj_bbox_info, obj_counts_info, class_remapper):
    for src_class, dst_classes in class_remapper.items():
        src_bbox_info = []
        src_counts_info = 0
        for dst_class in dst_classes:
            if dst_class in obj_bbox_info:
                src_bbox_info += obj_bbox_info[dst_class]
                src_counts_info += obj_counts_info[dst_class]

            obj_bbox_info.pop(dst_class, None)
            obj_counts_info.pop(dst_class, None)

        if src_counts_info > 0:
            obj_bbox_info[src_class] = src_bbox_info
            obj_counts_info[src_class] = src_counts_info

    return obj_bbox_info, obj_counts_info
