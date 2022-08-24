filepath = '/home/jonas/Documents/rotpot/potsdam_vs_xl_masters.md'
import re

def sub_timestamp(match):
    m, s = [int(x) for x in match.group(0).split(':')]
    link = f'https://www.youtube.com/watch?v=RAl0v3TtWWw&t={m}m{s}s'
    h, m = m//60 , m%60
    text = f'{m}:{s}'
    if h:
        text = f'{h}:' + text
    return f'[{text}]({link})'

if __name__ == '__main__':
    with open(filepath, 'r') as f:
        text = f.read()
    ts_regex = re.compile(r'\d+?:\d\d')
    text_subbed = ts_regex.sub(sub_timestamp, text)
    with open(filepath[:-3] + '_2.md', f'w') as f:
        f.write(text_subbed)

