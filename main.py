import requests

if __name__ == '__main__':
    responses = requests.post("https://api.c2smr.fr/machine/add_picture_alert_or_moment", files={
        'file':
            open('test.png', 'rb')
    })
    print(responses)
