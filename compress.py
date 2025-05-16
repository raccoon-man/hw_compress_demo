
def compress(template, original):
    bit_info = ''
    diff = ''
    byte_number = 0
    for elem,temp in zip(original, template):
        byte_number += 1
        if elem == ' ':
            bit_info += ' '
        # 和模板数据相同
        elif elem == temp and elem != ',':
            bit_info += '1'
        # 和模板数据不同
        elif elem != temp:
            bit_info += '0'
            diff += elem
    return bit_info, diff


def compress_block(template, data):
    compressed_data = []
    difference = []
    rows = data.apply(lambda row: ','.join(row), axis=1).tolist()
    for row in rows:
        data,diff = compress(template, row)
        compressed_data.append(data)
        difference.append(diff)
    return compressed_data, difference
