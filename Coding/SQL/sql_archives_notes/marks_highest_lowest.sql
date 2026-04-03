
drop table marks; 
CREATE TABLE marks(
	student_id varchar,
	marks INT NOT NULL,
	subject varchar NOT NULL,
	PRIMARY KEY(student_id,subject)
);

INSERT INTO marks(student_id, subject, marks) 
VALUES
	('a','english',82),
	('a','maths',53),
	('a','science',40),
	('b','english',87),
	('b','maths',25),
	('b','science',74),
	('c','english',35),
	('c','maths',54),
	('c','science',39)
;

with marks_highest_lowest as (
	select a.subject,max(a.marks) as highest_marks,min(a.marks) as lowest_marks
	from marks a group by a.subject 
), 
marks_mapping as (
	select a.*,b.student_id as highest_mark_student_id, c.student_id as highest_mark_student_id
	from marks_highest_lowest a 
	inner join marks b on b.marks = a.highest_marks and b.subject = a.subject
	inner join marks c on c.marks = a.lowest_marks and c.subject = a.subject
)
select * from marks_mapping; 

select * from marks ; 

