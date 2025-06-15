
def compress(template1, template2, original):
    bit_info = ''
    diff = ''
    byte_number = 0
    for elem, temp1, temp2 in zip(original, template1, template2):
        byte_number += 1
        # 和模板数据相同
        if elem == ' ':
            bit_info += ' '
        elif elem == ',':
            pass
        elif elem == temp1:
            bit_info += '1'
        elif elem == temp2:
            bit_info += '2'
        else:
            bit_info += '0'
            diff += elem
    return bit_info, diff


def compress_block(template1, template2, data):
    compressed_data = []
    difference = []
    rows = data.apply(lambda row: ','.join(row), axis=1).tolist()
    for row in rows:
        data,diff = compress(template1, template2, row)
        compressed_data.append(data)
        difference.append(diff)
    return compressed_data, difference
