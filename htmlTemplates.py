css = '''
<style>
.chat-message {
  display: flex; 
  margin-bottom: 10px;
  align-items: center; 
}

.chat-message {
  background-color: #f2f2f2; 
  border-radius: 5px;
  padding: 10px;
}

.chat-message.bot {
  flex-direction: row-reverse;
  background-color: #ddd;
}

.avatar {
  width: 50px;
  height: 50px;
  border-radius: 50%;
  margin: 5px;
}

.message-content {
  flex: 1;
  padding: 0 10px;
}

.message-content p {
  margin: 0;
  font-size: 16px;
}
.st-emotion-cache-vk3wp9 {
    height: 100%!important;
}
'''

user_template = """<div class="chat-message">
  <div class="avatar">
    <img width="100%" border-radius="50%" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTlYJDxIG6gDt8sQefCPFk4dZhD27AjdbeofP1t0RE-TQ&s" alt="User Avatar">
  </div>
  <div class="message-content">
    <p>{{MSG}}</p>
  </div>
</div>
"""

bot_template = """<div class="chat-message bot">
  <div class="avatar">
    <img width="100%" border-radius="50%" src="https://img.freepik.com/free-vector/cute-robot-wearing-hat-flying-cartoon-vector-icon-illustration-science-technology-icon-isolated_138676-5186.jpg?size=338&ext=jpg&ga=GA1.1.1224184972.1712102400&semt=ais" alt="Bot Avatar">
  </div>
  <div class="message-content">
    <p>{{MSG}}</p>
  </div>
</div>
"""
