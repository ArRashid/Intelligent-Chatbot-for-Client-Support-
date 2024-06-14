import requests
import json

api_key = "S6isdYJbZEnXwTs0VfMR"
domain = "ginesys"
password = "password"
agent_id = 33036115660
group_id = 33000214017


def GetTicketConversations(ticketid):
    r = requests.get("https://" + domain + '.freshdesk.com/api/v2/tickets/' +
                     str(ticketid)+'/conversations', auth=(api_key, password))
    if r.status_code == 200:
        data = json.loads(r.text)
    else:
        # the api request not returning 200
        data = "error"

    return {"STATUS": r.status_code, "DATA": data, "SOURCE": "https://" + domain + '.freshdesk.com/a/tickets/'+str(ticketid)}


# This funsation to send reply by providing ticket id and massage (in HTML format)
def CreateTicket(subject: str, description: str):
    payload = {
        "description": description,
        "subject": subject,
        "email": "abdur.m@gls.in",
        "status": 2,
        "priority": 1,
        "requester_id": agent_id,
        "group_id": group_id,
        "custom_fields": {"cf_application": 'Ginesys (ERP)', "cf_area_operation": 'Configuration/ Installation/Setup'}

    }
    r = requests.post("https://" + domain + '.freshdesk.com//api/v2/tickets',
                      auth=(api_key, password), json=payload)
    data = json.loads(r.text)
    print(data)
    return {"STATUS": r.status_code, "DATA": data, "TICKET": "https://" + domain + '.freshdesk.com/a/tickets/'+str(data['id'])}


def get_ticket_convasation(ticketid, fields_to_check: dict = None):
    TicketInfo = GetTicketConversations(ticketid)
    counter = 0
    All_body_texts = ""
    if fields_to_check != None:
        for data in TicketInfo["DATA"]:
            for field, value in fields_to_check.items():
                if data[field] == value and TicketInfo["STATUS"] == 200:
                    All_body_texts += f"Ticket Convasation No : {counter} \n Convasation Body :{data['body_text']} \n\n"
                    counter += 1
    else:
        if TicketInfo["STATUS"] == 200:
            for data in TicketInfo["DATA"]:
                print(data)
                All_body_texts += f"Ticket Convasation No : {counter} \n Convasation Body :{data['body_text']} \n\n"
                counter += 1
        else:
            print(
                f"Error the ticket id {ticketid} is unable to read by api. HTTP status code : {TicketInfo['STATUS']}")
    return All_body_texts, TicketInfo["SOURCE"], TicketInfo['STATUS']
