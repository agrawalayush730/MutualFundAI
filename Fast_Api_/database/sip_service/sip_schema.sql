CREATE TABLE sip_requests (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT, 
    amount VARCHAR(50) NOT NULL,
    fund_name VARCHAR(255) NOT NULL,
    duration VARCHAR(50),
    start_date VARCHAR(50),
    frequency VARCHAR(50),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);