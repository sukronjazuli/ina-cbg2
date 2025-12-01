import os

# Check if .env file exists
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        content = f.read()
        print('Env file content length:', len(content))
        print('Has GOOGLE_API_KEY:', 'GOOGLE_API_KEY' in content)
        if 'GOOGLE_API_KEY' in content:
            lines = content.split('\n')
            for line in lines:
                if line.startswith('GOOGLE_API_KEY'):
                    key_value = line.split('=', 1)[1] if '=' in line else ''
                    print('Key length:', len(key_value.strip()))
                    break
else:
    print('.env file not found')
