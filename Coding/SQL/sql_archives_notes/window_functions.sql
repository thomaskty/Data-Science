CREATE TABLE sales(
	year SMALLINT CHECK(year > 0),
	group_id INT NOT NULL,
	amount DECIMAL(10,2) NOT NULL,
	PRIMARY KEY(year,group_id)
);

INSERT INTO 
	sales(year, group_id, amount) 
VALUES
	(2018,1,1474),
	(2018,2,1787),
	(2018,3,1760),
	(2019,1,1915),
	(2019,2,1911),
	(2019,3,1118),
	(2020,1,1646),
	(2020,2,1975),
	(2020,3,1516)
;
