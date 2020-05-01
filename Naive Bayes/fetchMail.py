import email
import imaplib
import base64

username='fengx7897@gmail.com'
password='009778@fxy'

mail=imaplib.IMAP4_SSL("imap.gmail.com")
mail.login(username,password)

mail.select()
mail.list()

result,data=mail.search(None, 'ALL')


inbox_item_list=data[0].split()

most_recent=inbox_item_list[-1]

result2,email_data=mail.uid('fetch',most_recent,'(RFC822)')

raw_email=email_data[0][1].decode()

email_message=email.message_from_string(raw_email)

print(email_message['subject'].split())